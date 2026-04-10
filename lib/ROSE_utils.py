"""
ROSE_utils.py — Utility methods for ROSE pipeline (GPU-accelerated version)

Key changes from original:
  - Bam class uses pysam instead of subprocess calls to samtools
  - LocusCollection has GPU-accelerated overlap methods
  - All original interfaces are preserved for backward compatibility
"""

import os
import re
import subprocess
import datetime
import logging
import numpy as np
from collections import defaultdict

# Optional imports
try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

# GPU kernel imports (optional)
try:
    from ROSE_cuda_kernels import compute_overlap, compute_density
    from ROSE_gpu_utils import GPU_AVAILABLE
    HAS_GPU_KERNELS = True
except ImportError:
    HAS_GPU_KERNELS = False
    GPU_AVAILABLE = False

logger = logging.getLogger("ROSE_GPU")

# ==================================================================
# ==========================I/O FUNCTIONS===========================
# ==================================================================

def unParseTable(table, output, sep):
    fh_out = open(output, 'w')
    if len(sep) == 0:
        for i in table:
            fh_out.write(str(i))
            fh_out.write('\n')
    else:
        for line in table:
            line = [str(x) for x in line]
            line = sep.join(line)
            fh_out.write(line)
            fh_out.write('\n')
    fh_out.close()


def parseTable(fn, sep, header=False, excel=False):
    fh = open(fn)
    lines = fh.readlines()
    fh.close()
    if excel:
        lines = lines[0].split('\r')
    if lines[0].count('\r') > 0:
        lines = lines[0].split('\r')
    table = []
    if header == True:
        lines = lines[1:]
    for i in lines:
        table.append(i[:-1].split(sep))
    return table


def bedToGFF(bed, output=''):
    '''turns a bed into a gff file'''
    if type(bed) == str:
        bed = parseTable(bed, '\t')
    gff = []
    for line in bed:
        gffLine = [line[0], line[3], '', line[1], line[2], line[4], '.', '', line[3]]
        gff.append(gffLine)
    if len(output) > 0:
        unParseTable(gff, output, '\t')
    else:
        return gff


def gffToBed(gff, output=''):
    '''turns a gff to a bed file'''
    bed = []
    for line in gff:
        newLine = [line[0], line[3], line[4], line[1], 0, line[6]]
        bed.append(newLine)
    if len(output) == 0:
        return bed
    else:
        unParseTable(bed, output, '\t')


def formatFolder(folderName, create=False):
    '''makes sure a folder exists and if not makes it'''
    if folderName[-1] != '/':
        folderName += '/'
    try:
        foo = os.listdir(folderName)
        return folderName
    except OSError:
        print(('folder %s does not exist' % (folderName)))
        if create:
            os.makedirs(folderName, exist_ok=True)
            return folderName
        else:
            return False


# ==================================================================
# ===================ANNOTATION FUNCTIONS===========================
# ==================================================================

def makeStartDict(annotFile, geneList=[]):
    '''
    makes a dictionary keyed by refseq ID that contains information about
    chrom/start/stop/strand/common name
    '''
    if type(geneList) == str:
        geneList = parseTable(geneList, '\t')
        geneList = [line[0] for line in geneList]

    if annotFile.upper().count('REFSEQ') >= 0:
        refseqTable, refseqDict = importRefseq(annotFile)
        if len(geneList) == 0:
            geneList = list(refseqDict.keys())
        startDict = {}
        for gene in geneList:
            if (gene in refseqDict) == False:
                continue
            startDict[gene] = {}
            startDict[gene]['sense'] = refseqTable[refseqDict[gene][0]][3]
            startDict[gene]['chr'] = refseqTable[refseqDict[gene][0]][2]
            startDict[gene]['start'] = getTSSs([gene], refseqTable, refseqDict)
            if startDict[gene]['sense'] == '+':
                startDict[gene]['end'] = [int(refseqTable[refseqDict[gene][0]][5])]
            else:
                startDict[gene]['end'] = [int(refseqTable[refseqDict[gene][0]][4])]
            startDict[gene]['name'] = refseqTable[refseqDict[gene][0]][12]
    return startDict


def getTSSs(geneList, refseqTable, refseqDict):
    if len(geneList) == 0:
        refseq = refseqTable
    else:
        refseq = refseqFromKey(geneList, refseqDict, refseqTable)
    TSS = []
    for line in refseq:
        if line[3] == '+':
            TSS.append(line[4])
        if line[3] == '-':
            TSS.append(line[5])
    TSS = list(map(int, TSS))
    return TSS


def refseqFromKey(refseqKeyList, refseqDict, refseqTable):
    typeRefseq = []
    for name in refseqKeyList:
        if name in refseqDict:
            typeRefseq.append(refseqTable[refseqDict[name][0]])
    return typeRefseq


def importRefseq(refseqFile, returnMultiples=False):
    '''opens up a refseq file downloaded by UCSC'''
    refseqTable = parseTable(refseqFile, '\t')
    refseqDict = {}
    ticker = 1
    for line in refseqTable[1:]:
        if line[1] in refseqDict:
            refseqDict[line[1]].append(ticker)
        else:
            refseqDict[line[1]] = [ticker]
        ticker = ticker + 1
    multiples = []
    for i in refseqDict:
        if len(refseqDict[i]) > 1:
            multiples.append(i)
    if returnMultiples == True:
        return refseqTable, refseqDict, multiples
    else:
        return refseqTable, refseqDict


# ==================================================================
# ========================LOCUS INSTANCE============================
# ==================================================================

class Locus:
    __chrDict = dict()
    __senseDict = {'+': '+', '-': '-', '.': '.'}

    def __init__(self, chr, start, end, sense, ID=''):
        coords = [int(start), int(end)]
        coords.sort()
        if not (chr in self.__chrDict):
            self.__chrDict[chr] = chr
        self._chr = self.__chrDict[chr]
        self._sense = self.__senseDict[sense]
        self._start = int(coords[0])
        self._end = int(coords[1])
        self._ID = ID

    def ID(self): return self._ID
    def chr(self): return self._chr
    def start(self): return self._start
    def end(self): return self._end
    def len(self): return self._end - self._start + 1

    def getAntisenseLocus(self):
        if self._sense == '.':
            return self
        else:
            switch = {'+': '-', '-': '+'}
            return Locus(self._chr, self._start, self._end, switch[self._sense])

    def coords(self): return [self._start, self._end]
    def sense(self): return self._sense

    def overlaps(self, otherLocus):
        if self.chr() != otherLocus.chr(): return False
        elif not (self._sense == '.' or
                  otherLocus.sense() == '.' or
                  self.sense() == otherLocus.sense()): return False
        elif self.start() > otherLocus.end() or otherLocus.start() > self.end(): return False
        else: return True

    def contains(self, otherLocus):
        if self.chr() != otherLocus.chr(): return False
        elif not (self._sense == '.' or
                  otherLocus.sense() == '.' or
                  self.sense() == otherLocus.sense()): return False
        elif self.start() > otherLocus.start() or otherLocus.end() > self.end(): return False
        else: return True

    def overlapsAntisense(self, otherLocus):
        return self.getAntisenseLocus().overlaps(otherLocus)

    def containsAntisense(self, otherLocus):
        return self.getAntisenseLocus().contains(otherLocus)

    def __hash__(self): return self._start + self._end
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        if self.chr() != other.chr(): return False
        if self.start() != other.start(): return False
        if self.end() != other.end(): return False
        if self.sense() != other.sense(): return False
        return True
    def __ne__(self, other): return not (self.__eq__(other))
    def __str__(self): return self.chr() + '(' + self.sense() + '):' + '-'.join(map(str, self.coords()))

    def checkRep(self):
        pass


class LocusCollection:
    def __init__(self, loci, windowSize):
        self.__chrToCoordToLoci = dict()
        self.__loci = dict()
        self.__winSize = windowSize
        for lcs in loci:
            self.__addLocus(lcs)

    def __addLocus(self, lcs):
        if not (lcs in self.__loci):
            self.__loci[lcs] = None
            if lcs.sense() == '.':
                chrKeyList = [lcs.chr() + '+', lcs.chr() + '-']
            else:
                chrKeyList = [lcs.chr() + lcs.sense()]
            for chrKey in chrKeyList:
                if not (chrKey in self.__chrToCoordToLoci):
                    self.__chrToCoordToLoci[chrKey] = dict()
                for n in self.__getKeyRange(lcs):
                    if not (n in self.__chrToCoordToLoci[chrKey]):
                        self.__chrToCoordToLoci[chrKey][n] = []
                    self.__chrToCoordToLoci[chrKey][n].append(lcs)

    def __getKeyRange(self, locus):
        start = locus.start() // self.__winSize
        end = locus.end() // self.__winSize + 1
        return range(start, end)

    def __len__(self): return len(self.__loci)

    def append(self, new): self.__addLocus(new)
    def extend(self, newList):
        for lcs in newList: self.__addLocus(lcs)

    def hasLocus(self, locus):
        return locus in self.__loci

    def remove(self, old):
        if not (old in self.__loci):
            raise ValueError("requested locus isn't in collection")
        del self.__loci[old]
        if old.sense() == '.':
            senseList = ['+', '-']
        else:
            senseList = [old.sense()]
        for k in self.__getKeyRange(old):
            for sense in senseList:
                self.__chrToCoordToLoci[old.chr() + sense][k].remove(old)

    def getWindowSize(self): return self.__winSize
    def getLoci(self): return list(self.__loci.keys())

    def getChrList(self):
        tempKeys = dict()
        for k in list(self.__chrToCoordToLoci.keys()):
            tempKeys[k[:-1]] = None
        return list(tempKeys.keys())

    def __subsetHelper(self, locus, sense):
        sense = sense.lower()
        if ['sense', 'antisense', 'both'].count(sense) != 1:
            raise ValueError("sense command invalid: '" + sense + "'.")
        matches = dict()
        senses = ['+', '-']
        if locus.sense() == '.' or sense == 'both':
            lamb = lambda s: True
        elif sense == 'sense':
            lamb = lambda s: s == locus.sense()
        elif sense == 'antisense':
            lamb = lambda s: s != locus.sense()
        else:
            raise ValueError("sense value was inappropriate: '" + sense + "'.")
        for s in filter(lamb, senses):
            chrKey = locus.chr() + s
            if chrKey in self.__chrToCoordToLoci:
                for n in self.__getKeyRange(locus):
                    if n in self.__chrToCoordToLoci[chrKey]:
                        for lcs in self.__chrToCoordToLoci[chrKey][n]:
                            matches[lcs] = None
        return list(matches.keys())

    def getOverlap(self, locus, sense='sense'):
        matches = self.__subsetHelper(locus, sense)
        realMatches = dict()
        if sense == 'sense' or sense == 'both':
            for i in [lcs for lcs in matches if lcs.overlaps(locus)]:
                realMatches[i] = None
        if sense == 'antisense' or sense == 'both':
            for i in [lcs for lcs in matches if lcs.overlapsAntisense(locus)]:
                realMatches[i] = None
        return list(realMatches.keys())

    def getContained(self, locus, sense='sense'):
        matches = self.__subsetHelper(locus, sense)
        realMatches = dict()
        if sense == 'sense' or sense == 'both':
            for i in [lcs for lcs in matches if locus.contains(lcs)]:
                realMatches[i] = None
        if sense == 'antisense' or sense == 'both':
            for i in [lcs for lcs in matches if locus.containsAntisense(lcs)]:
                realMatches[i] = None
        return list(realMatches.keys())

    def getContainers(self, locus, sense='sense'):
        matches = self.__subsetHelper(locus, sense)
        realMatches = dict()
        if sense == 'sense' or sense == 'both':
            for i in [lcs for lcs in matches if lcs.contains(locus)]:
                realMatches[i] = None
        if sense == 'antisense' or sense == 'both':
            for i in [lcs for lcs in matches if lcs.containsAntisense(locus)]:
                realMatches[i] = None
        return list(realMatches.keys())

    def stitchCollection(self, stitchWindow=1, sense='both'):
        '''reduces the collection by stitching together overlapping loci'''
        locusList = self.getLoci()
        oldCollection = LocusCollection(locusList, 500)
        stitchedCollection = LocusCollection([], 500)

        for locus in locusList:
            if oldCollection.hasLocus(locus):
                oldCollection.remove(locus)
                overlappingLoci = oldCollection.getOverlap(
                    Locus(locus.chr(), locus.start() - stitchWindow,
                          locus.end() + stitchWindow, locus.sense(), locus.ID()), sense)

                stitchTicker = 1
                while len(overlappingLoci) > 0:
                    stitchTicker += len(overlappingLoci)
                    overlapCoords = locus.coords()
                    for overlappingLocus in overlappingLoci:
                        overlapCoords += overlappingLocus.coords()
                        oldCollection.remove(overlappingLocus)
                    if sense == 'both':
                        locus = Locus(locus.chr(), min(overlapCoords), max(overlapCoords), '.', locus.ID())
                    else:
                        locus = Locus(locus.chr(), min(overlapCoords), max(overlapCoords), locus.sense(), locus.ID())
                    overlappingLoci = oldCollection.getOverlap(
                        Locus(locus.chr(), locus.start() - stitchWindow,
                              locus.end() + stitchWindow, locus.sense()), sense)
                locus._ID = '%s_%s_lociStitched' % (stitchTicker, locus.ID())
                stitchedCollection.append(locus)
            else:
                continue
        return stitchedCollection

    # ---------------------------------------------------------------
    # GPU-accelerated batch overlap (NEW)
    # ---------------------------------------------------------------

    def getBatchOverlapGPU(self, queryLoci, sense='both', use_gpu=True):
        """
        GPU-accelerated batch overlap: find overlapping loci for multiple queries at once.

        Args:
            queryLoci: list of Locus objects to query
            sense: 'sense', 'antisense', or 'both'
            use_gpu: whether to attempt GPU acceleration

        Returns:
            list of lists: for each query, list of overlapping Locus objects
        """
        if not queryLoci:
            return []

        # For small queries or no GPU, fall back to standard method
        if len(queryLoci) < 50 or not HAS_GPU_KERNELS or not use_gpu:
            return [self.getOverlap(q, sense) for q in queryLoci]

        # Group queries by chromosome for efficient GPU processing
        chr_groups = defaultdict(list)
        for i, q in enumerate(queryLoci):
            chr_groups[q.chr()].append((i, q))

        results = [[] for _ in range(len(queryLoci))]

        for chrom, indexed_queries in chr_groups.items():
            # Get target loci on this chromosome
            targets = []
            for lcs in self.getLoci():
                if lcs.chr() == chrom:
                    targets.append(lcs)

            if not targets:
                continue

            # Sort targets by start
            targets.sort(key=lambda x: x.start())

            # Build arrays
            q_starts = np.array([q.start() for _, q in indexed_queries], dtype=np.int32)
            q_ends = np.array([q.end() for _, q in indexed_queries], dtype=np.int32)
            t_starts = np.array([t.start() for t in targets], dtype=np.int32)
            t_ends = np.array([t.end() for t in targets], dtype=np.int32)

            # GPU overlap
            counts, sizes, indices = compute_overlap(q_starts, q_ends, t_starts, t_ends, use_gpu=use_gpu)

            # Use the extracted indices natively
            offset = 0
            for qi, (orig_idx, query) in enumerate(indexed_queries):
                c = counts[qi]
                if c > 0:
                    query_targets = [targets[indices[offset + j]] for j in range(c)]
                    # Optional: filter by sense if needed, but since batch_interval_overlap
                    # just checks coordinate overlap, we must post-filter for sense here if sense != 'both'
                    if sense == 'both':
                        results[orig_idx] = query_targets
                    elif sense == 'sense':
                        results[orig_idx] = [t for t in query_targets if t.sense() == '.' or query.sense() == '.' or t.sense() == query.sense()]
                    elif sense == 'antisense':
                        results[orig_idx] = [t for t in query_targets if t.sense() == '.' or query.sense() == '.' or t.sense() != query.sense()]
                    
                    offset += c

        return results


# ==================================================================
# ========================LOCUS FUNCTIONS===========================
# ==================================================================

def locusCollectionToGFF(locusCollection):
    lociList = locusCollection.getLoci()
    gff = []
    for locus in lociList:
        newLine = [locus.chr(), locus.ID(), '', locus.coords()[0], locus.coords()[1], '', locus.sense(), '', locus.ID()]
        gff.append(newLine)
    return gff


def gffToLocusCollection(gff, window=500):
    '''opens up a gff file and turns it into a LocusCollection instance'''
    lociList = []
    if type(gff) == str:
        gff = parseTable(gff, '\t')
    for line in gff:
        if len(line[2]) > 0:
            name = line[2]
        elif len(line[8]) > 0:
            name = line[8]
        else:
            name = '%s:%s:%s-%s' % (line[0], line[6], line[3], line[4])
        lociList.append(Locus(line[0], line[3], line[4], line[6], name))
    return LocusCollection(lociList, window)


def makeTranscriptCollection(annotFile, upSearch, downSearch, window=500, geneList=[]):
    '''makes a LocusCollection w/ each transcript as a locus'''
    if annotFile.upper().count('REFSEQ') >= 0:
        refseqTable, refseqDict = importRefseq(annotFile)
        locusList = []
        ticker = 0
        if len(geneList) == 0:
            geneListSet = set(refseqDict.keys())
        else:
            geneListSet = set(geneList)
            
        for line in refseqTable[1:]:
            if line[1] in geneListSet:
                if line[3] == '-':
                    locus = Locus(line[2], int(line[4]) - downSearch, int(line[5]) + upSearch, line[3], line[1])
                else:
                    locus = Locus(line[2], int(line[4]) - upSearch, int(line[5]) + downSearch, line[3], line[1])
                locusList.append(locus)
                ticker = ticker + 1
                if ticker % 5000 == 0:
                    pass
    transCollection = LocusCollection(locusList, window)
    return transCollection


def makeTSSLocus(gene, startDict, upstream, downstream):
    '''given a startDict, make a locus for any gene's TSS'''
    start = startDict[gene]['start'][0]
    if startDict[gene]['sense'] == '-':
        return Locus(startDict[gene]['chr'], start - downstream, start + upstream, '-', gene)
    else:
        return Locus(startDict[gene]['chr'], start - upstream, start + downstream, '+', gene)


def makeSearchLocus(locus, upSearch, downSearch):
    if locus.sense() == '-':
        searchLocus = Locus(locus.chr(), locus.start() - downSearch, locus.end() + upSearch, locus.sense(), locus.ID())
    else:
        searchLocus = Locus(locus.chr(), locus.start() - upSearch, locus.end() + downSearch, locus.sense(), locus.ID())
    return searchLocus


# ==================================================================
# ==========================BAM CLASS===============================
# ==================================================================

def checkChrStatus(bamFile):
    """Check if BAM file uses 'chr' prefix in chromosome names."""
    if HAS_PYSAM:
        try:
            with pysam.AlignmentFile(bamFile, "rb") as bam:
                for read in bam.head(1):
                    chrom = read.reference_name
                    if chrom and chrom.startswith('chr'):
                        return 1
                    else:
                        return 0
            return 0
        except Exception:
            pass

    # Fallback to samtools
    command = 'samtools view %s | head -n 1' % (bamFile)
    stats = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE, shell=True)
    statLines = stats.stdout.readlines()
    stats.stdout.close()
    chrPattern = re.compile('chr')
    for line in statLines:
        line = line.decode("utf-8")
        sline = line.split("\t")
        if re.search(chrPattern, sline[2]):
            return 1
        else:
            return 0
    return 0


def convertBitwiseFlag(flag):
    if int(flag) & 16:
        return "-"
    else:
        return "+"


class Bam:
    '''A class for a sorted and indexed bam file — uses pysam when available'''

    def __init__(self, bamFile):
        self._bam = bamFile
        self._pysam_handle = None
        self._use_pysam = HAS_PYSAM

        if self._use_pysam:
            try:
                self._pysam_handle = pysam.AlignmentFile(bamFile, "rb")
            except Exception as e:
                logger.warning(f"pysam failed to open {bamFile}: {e}. Falling back to samtools.")
                self._use_pysam = False

    def __del__(self):
        if self._pysam_handle is not None:
            try:
                self._pysam_handle.close()
            except Exception:
                pass

    def getTotalReads(self, readType='mapped'):
        if self._use_pysam:
            try:
                stats = self._pysam_handle.get_index_statistics()
                total = sum(s.mapped for s in stats)
                if readType == 'mapped':
                    return total
                elif readType == 'total':
                    return total + sum(s.unmapped for s in stats)
            except Exception:
                pass

        # Fallback to samtools
        command = 'samtools flagstat %s' % (self._bam)
        stats = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                 stdout=subprocess.PIPE, shell=True)
        statLines = stats.stdout.readlines()
        stats.stdout.close()
        if readType == 'mapped':
            for line in statLines:
                line = line.decode("utf-8")
                if line.count('mapped (') == 1:
                    return int(line.split(' ')[0])
        if readType == 'total':
            return int(statLines[0].decode("utf-8").split(' ')[0])

    def convertBitwiseFlag(self, flag):
        if flag & 16:
            return "-"
        else:
            return "+"

    def getRawReads(self, locus, sense, unique=False, includeJxnReads=False, printCommand=False):
        '''gets raw reads from the bam — uses pysam when available.'''

        if self._use_pysam:
            return self._getRawReadsPysam(locus, sense, unique, includeJxnReads)

        # Original samtools fallback
        locusLine = locus.chr() + ':' + str(locus.start()) + '-' + str(locus.end())
        command = 'samtools view %s %s' % (self._bam, locusLine)
        if printCommand:
            print(command)
        getReads = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE, shell=True)
        reads = getReads.communicate()
        reads = reads[0].decode("utf-8")
        reads = reads.split('\n')[:-1]
        reads = [read.split('\t') for read in reads]
        if includeJxnReads == False:
            reads = [x for x in reads if x[5].count('N') < 1]
        convertDict = {'16': '-', '0': '+', '64': '+', '65': '+', '80': '-', '81': '-',
                        '129': '+', '145': '-', '256': '+', '272': '-', '99': '+', '147': '-'}
        keptReads = []
        seqDict = defaultdict(int)
        if sense == '-':
            strand = ['+', '-']
            strand.remove(locus.sense())
            strand = strand[0]
        else:
            strand = locus.sense()
        for read in reads:
            readStrand = convertBitwiseFlag(read[1])
            if sense == 'both' or sense == '.' or readStrand == strand:
                if unique and seqDict[read[9]] == 0:
                    keptReads.append(read)
                elif not unique:
                    keptReads.append(read)
            seqDict[read[9]] += 1
        return keptReads

    def _getRawReadsPysam(self, locus, sense, unique=False, includeJxnReads=False):
        '''Get raw reads using pysam — much faster than subprocess.'''
        chrom = locus.chr()
        start = locus.start()
        end = locus.end()

        # IMPORTANT: samtools view uses 1-based inclusive coordinates (chr:start-end)
        # pysam.fetch uses 0-based half-open coordinates [start, end)
        # To match samtools: fetch(chrom, start-1, end)
        # so that 0-based [start-1, end) = 1-based [start, end]
        fetch_start = max(0, start - 1)
        fetch_end = end

        try:
            reads_iter = self._pysam_handle.fetch(chrom, fetch_start, fetch_end)
        except ValueError:
            # Try without 'chr' prefix or with it
            if chrom.startswith('chr'):
                try:
                    reads_iter = self._pysam_handle.fetch(chrom[3:], fetch_start, fetch_end)
                except Exception:
                    return []
            else:
                try:
                    reads_iter = self._pysam_handle.fetch('chr' + chrom, fetch_start, fetch_end)
                except Exception:
                    return []

        keptReads = []
        seqDict = defaultdict(int)

        if sense == '-':
            strand = ['+', '-']
            strand.remove(locus.sense())
            strand = strand[0]
        else:
            strand = locus.sense()

        for read in reads_iter:
            # Skip junction reads if requested
            if not includeJxnReads and read.cigarstring and 'N' in read.cigarstring:
                continue

            readStrand = '-' if read.is_reverse else '+'
            seq = read.query_sequence or ''

            if sense == 'both' or sense == '.' or readStrand == strand:
                if unique and seqDict[seq] > 0:
                    continue
                # Format as list matching samtools view output format
                read_data = [
                    read.query_name,                    # 0: QNAME
                    str(read.flag),                     # 1: FLAG
                    read.reference_name,                # 2: RNAME
                    str(read.reference_start + 1),      # 3: POS (1-based)
                    str(read.mapping_quality),           # 4: MAPQ
                    read.cigarstring or '*',             # 5: CIGAR
                    '*',                                 # 6: RNEXT
                    '0',                                 # 7: PNEXT
                    '0',                                 # 8: TLEN
                    seq,                                 # 9: SEQ
                    read.qual or '*'                     # 10: QUAL
                ]
                keptReads.append(read_data)
                seqDict[seq] += 1

        return keptReads

    def readsToLoci(self, reads, IDtag='sequence,seqID,none'):
        '''takes raw read lines from the bam and converts them into loci'''
        loci = []
        ID = ''
        if IDtag == 'sequence,seqID,none':
            print('please specify one of the three options: sequence, seqID, none')
            return
        numPattern = re.compile(r'\d*')
        for read in reads:
            chrom = read[2]
            strand = convertBitwiseFlag(read[1])
            if IDtag == 'sequence':
                ID = read[9]
            elif IDtag == 'seqID':
                ID = read[0]
            else:
                ID = ''
            length = len(read[9])
            start = int(read[3])
            if read[5].count('N') == 1:
                [first, gap, second] = [int(x) for x in [x for x in re.findall(numPattern, read[5]) if len(x) > 0]][0:3]
                if IDtag == 'sequence':
                    loci.append(Locus(chrom, start, start + first, strand, ID[0:first]))
                    loci.append(Locus(chrom, start + first + gap, start + first + gap + second, strand, ID[first:]))
                else:
                    loci.append(Locus(chrom, start, start + first, strand, ID))
                    loci.append(Locus(chrom, start + first + gap, start + first + gap + second, strand, ID))
            elif read[5].count('N') > 1:
                continue
            else:
                loci.append(Locus(chrom, start, start + length, strand, ID))
        return loci

    def getReadsLocus(self, locus, sense='both', unique=True, IDtag='sequence,seqID,none', includeJxnReads=False):
        '''gets all of the reads for a given locus'''
        reads = self.getRawReads(locus, sense, unique, includeJxnReads)
        loci = self.readsToLoci(reads, IDtag)
        return loci

    def getReadsAsArrays(self, locus, extension=200, sense='both'):
        """
        NEW: Get reads as numpy arrays of start/end positions.
        Optimized for GPU density computation — avoids creating Locus objects.

        Returns:
            (starts, ends) — numpy int32 arrays of extended read positions
        """
        if self._use_pysam:
            return self._getReadsAsArraysPysam(locus, extension, sense)

        # Fallback: use getReadsLocus then convert
        reads = self.getRawReads(locus, sense, unique=False)
        if not reads:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        starts = []
        ends = []
        for read in reads:
            strand = convertBitwiseFlag(read[1])
            pos = int(read[3])
            length = len(read[9])
            if strand == '+' or strand == '.':
                starts.append(pos)
                ends.append(pos + length + extension)
            else:
                starts.append(pos - extension)
                ends.append(pos + length)

        return np.array(starts, dtype=np.int32), np.array(ends, dtype=np.int32)

    def _getReadsAsArraysPysam(self, locus, extension=200, sense='both'):
        """pysam-based fast array extraction of read positions."""
        chrom = locus.chr()
        start = locus.start()
        end = locus.end()

        try:
            reads_iter = self._pysam_handle.fetch(chrom, max(0, start - extension), end + extension)
        except ValueError:
            if chrom.startswith('chr'):
                try:
                    reads_iter = self._pysam_handle.fetch(chrom[3:], max(0, start - extension), end + extension)
                except Exception:
                    return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            else:
                try:
                    reads_iter = self._pysam_handle.fetch('chr' + chrom, max(0, start - extension), end + extension)
                except Exception:
                    return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        starts_list = []
        ends_list = []

        for read in reads_iter:
            if read.is_unmapped:
                continue
            read_start = read.reference_start
            read_end = read.reference_end or (read_start + read.query_length)

            # Extend reads
            if read.is_reverse:
                starts_list.append(read_start - extension)
                ends_list.append(read_end)
            else:
                starts_list.append(read_start)
                ends_list.append(read_end + extension)

        if not starts_list:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        return np.array(starts_list, dtype=np.int32), np.array(ends_list, dtype=np.int32)

    def getReadSequences(self, locus, sense='both', unique=True, includeJxnReads=False):
        reads = self.getRawReads(locus, sense, unique, includeJxnReads)
        return [read[9] for read in reads]

    def getReadStarts(self, locus, sense='both', unique=False, includeJxnReads=False):
        reads = self.getRawReads(locus, sense, unique, includeJxnReads)
        return [int(read[3]) for read in reads]

    def getReadCount(self, locus, sense='both', unique=True, includeJxnReads=False):
        reads = self.getRawReads(locus, sense, unique, includeJxnReads)
        return len(reads)


# ==================================================================
# ========================MISC FUNCTIONS============================
# ==================================================================

def uniquify(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def order(x, NoneIsLast=True, decreasing=False):
    """Returns the ordering of elements of x."""
    omitNone = False
    if NoneIsLast == None:
        NoneIsLast = True
        omitNone = True
    n = len(x)
    ix = list(range(n))
    if None not in x:
        ix.sort(reverse=decreasing, key=lambda j: x[j])
    else:
        def key(i, x=x):
            elem = x[i]
            if decreasing == NoneIsLast:
                return not (elem is None), elem
            else:
                return elem is None, elem
        ix = list(range(n))
        ix.sort(key=key, reverse=decreasing)
    if omitNone:
        n = len(x)
        for i in range(n - 1, -1, -1):
            if x[ix[i]] == None:
                n -= 1
        return ix[:n]
    return ix
