#!/usr/bin/env python3
"""
ROSE_main.py — Main ROSE pipeline orchestrator (GPU-accelerated version)

Stitches together regions to form enhancers, maps read density to stitched regions,
and ranks enhancers by read density to discover super-enhancers.

Key changes from original:
  - Direct Python calls instead of os.system() for BAM mapping
  - Multi-GPU support via ROSE_gpu_utils
  - Concurrent BAM mapping with ThreadPoolExecutor instead of background processes
  - pysam-based BAM I/O instead of subprocess samtools
  - Dynamic batch sizing based on available GPU memory
"""

import sys
import os
import time
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import ROSE_utils

# GPU imports (optional)
try:
    from ROSE_gpu_utils import (
        GPU_AVAILABLE, create_gpu_config, print_system_summary
    )
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    GPU_AVAILABLE = False

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s — %(message)s'
)
logger = logging.getLogger("ROSE_MAIN")


# ==================================================================
# =====================REGION STITCHING=============================
# ==================================================================

def regionStitching(inputGFF, stitchWindow, tssWindow, annotFile, removeTSS=True):
    print('PERFORMING REGION STITCHING')
    boundCollection = ROSE_utils.gffToLocusCollection(inputGFF)

    debugOutput = []
    if removeTSS:
        startDict = ROSE_utils.makeStartDict(annotFile)

        removeTicker = 0
        tssLoci = []
        for geneID in list(startDict.keys()):
            tssLoci.append(ROSE_utils.makeTSSLocus(geneID, startDict, tssWindow, tssWindow))

        tssCollection = ROSE_utils.LocusCollection(tssLoci, 50)
        boundLoci = boundCollection.getLoci()

        print('COMPUTING BATCH OVERLAPS FOR CONTAINERS')
        overlappingLociBatch = tssCollection.getBatchOverlapGPU(boundLoci, 'both', use_gpu=True)

        for i, locus in enumerate(boundLoci):
            overlappingTSS = overlappingLociBatch[i]
            if len(overlappingTSS) > 0:
                # A container must first be an overlap
                containers = [tss for tss in overlappingTSS if tss.contains(locus)]
                if len(containers) > 0:
                    boundCollection.remove(locus)
                    debugOutput.append([locus.__str__(), locus.ID(), 'CONTAINED'])
                    removeTicker += 1
        print(('REMOVED %s LOCI BECAUSE THEY WERE CONTAINED BY A TSS' % (removeTicker)))

    stitchedCollection = boundCollection.stitchCollection(stitchWindow, 'both')

    if removeTSS:
        fixedLoci = []
        tssLoci = []
        for geneID in list(startDict.keys()):
            tssLoci.append(ROSE_utils.makeTSSLocus(geneID, startDict, 50, 50))

        tssCollection = ROSE_utils.LocusCollection(tssLoci, 50)
        removeTicker = 0
        originalTicker = 0
        
        stitchedLoci = stitchedCollection.getLoci()
        print('COMPUTING BATCH OVERLAPS FOR MULTIPLE TSS')
        overlappingTSSLociBatch = tssCollection.getBatchOverlapGPU(stitchedLoci, 'both', use_gpu=True)
        originalLociBatch = boundCollection.getBatchOverlapGPU(stitchedLoci, 'both', use_gpu=True)
        
        for i, stitchedLocus in enumerate(stitchedLoci):
            overlappingTSSLoci = overlappingTSSLociBatch[i]
            tssNames = [startDict[tssLocus.ID()]['name'] for tssLocus in overlappingTSSLoci]
            tssNames = ROSE_utils.uniquify(tssNames)
            if len(tssNames) > 2:
                originalLoci = originalLociBatch[i]
                originalTicker += len(originalLoci)
                fixedLoci += originalLoci
                debugOutput.append([stitchedLocus.__str__(), stitchedLocus.ID(), 'MULTIPLE_TSS'])
                removeTicker += 1
            else:
                fixedLoci.append(stitchedLocus)

        print(('REMOVED %s STITCHED LOCI BECAUSE THEY OVERLAPPED MULTIPLE TSSs' % (removeTicker)))
        print(('ADDED BACK %s ORIGINAL LOCI' % (originalTicker)))
        fixedCollection = ROSE_utils.LocusCollection(fixedLoci, 50)
        return fixedCollection, debugOutput
    else:
        return stitchedCollection, debugOutput


# ==================================================================
# =====================REGION LINKING MAPPING=======================
# ==================================================================

def mapCollection(stitchedCollection, referenceCollection, bamFileList, mappedFolder, output, refName):
    '''makes a table of factor density in a stitched locus and ranks table by number of loci stitched together'''
    print('FORMATTING TABLE')
    loci = stitchedCollection.getLoci()

    locusTable = [['REGION_ID', 'CHROM', 'START', 'STOP', 'NUM_LOCI', 'CONSTITUENT_SIZE']]

    lociLenList = []

    for locus in list(loci):
        if locus.chr() == 'chrY':
            loci.remove(locus)

    for locus in loci:
        lociLenList.append(locus.len())
    lenOrder = ROSE_utils.order(lociLenList, decreasing=True)

    ticker = 0
    for i in lenOrder:
        ticker += 1
        if ticker % 1000 == 0:
            print(ticker)
        locus = loci[i]

        refEnrichSize = 0
        refOverlappingLoci = referenceCollection.getOverlap(locus, 'both')
        for refLocus in refOverlappingLoci:
            refEnrichSize += refLocus.len()

        try:
            stitchCount = int(locus.ID().split('_')[0])
        except ValueError:
            stitchCount = 1

        locusTable.append([locus.ID(), locus.chr(), locus.start(), locus.end(), stitchCount, refEnrichSize])

    print('GETTING MAPPED DATA')
    for bamFile in bamFileList:
        bamFileName = bamFile.split('/')[-1]

        print(('GETTING MAPPING DATA FOR  %s' % bamFile))
        print(('OPENING %s%s_%s_MAPPED.gff' % (mappedFolder, refName, bamFileName)))

        mappedGFF = ROSE_utils.parseTable('%s%s_%s_MAPPED.gff' % (mappedFolder, refName, bamFileName), '\t')

        signalDict = defaultdict(float)
        print(('MAKING SIGNAL DICT FOR %s' % (bamFile)))
        mappedLoci = []
        for line in mappedGFF[1:]:
            chrom = line[1].split('(')[0]
            start = int(line[1].split(':')[-1].split('-')[0])
            end = int(line[1].split(':')[-1].split('-')[1])
            mappedLoci.append(ROSE_utils.Locus(chrom, start, end, '.', line[0]))
            try:
                signalDict[line[0]] = float(line[2]) * (abs(end - start))
            except ValueError:
                print('WARNING NO SIGNAL FOR LINE:')
                print(line)
                continue

        mappedCollection = ROSE_utils.LocusCollection(mappedLoci, 500)
        locusTable[0].append(bamFileName)

        for i in range(1, len(locusTable)):
            signal = 0.0
            line = locusTable[i]
            lineLocus = ROSE_utils.Locus(line[1], line[2], line[3], '.')
            overlappingRegions = mappedCollection.getOverlap(lineLocus, sense='both')
            for region in overlappingRegions:
                signal += signalDict[region.ID()]
            locusTable[i].append(signal)

    ROSE_utils.unParseTable(locusTable, output, '\t')


# ==================================================================
# =====================BAM MAPPING FUNCTION=========================
# ==================================================================

def run_bam_mapping(bamFile, gffFile, mappedOut, nBin, extension, rpm, sense, use_gpu=True):
    """
    Run ROSE_bamToGFF mapping as a direct Python call instead of os.system().
    This avoids the need for polling/sleeping to detect output files.
    """
    import ROSE_bamToGFF  # Import here to avoid circular imports

    print(f"  Mapping: {bamFile.split('/')[-1]} → {mappedOut.split('/')[-1]}")
    t_start = time.time()

    newGFF = ROSE_bamToGFF.mapBamToGFF(
        bamFile, gffFile,
        sense=sense,
        extension=extension,
        floor=1,
        rpm=rpm,
        matrix=nBin,
        use_gpu=use_gpu
    )

    ROSE_utils.unParseTable(newGFF, mappedOut, '\t')

    elapsed = time.time() - t_start
    print(f"  ✅ Mapped {mappedOut.split('/')[-1]} in {elapsed:.1f}s")
    return mappedOut


# ==================================================================
# =========================MAIN METHOD==============================
# ==================================================================

def main():
    '''main run call'''
    debug = False

    from optparse import OptionParser
    usage = "usage: %prog [options] -g [GENOME] -i [INPUT_REGION_GFF] -r [RANKBY_BAM_FILE] -o [OUTPUT_FOLDER] [OPTIONAL_FLAGS]"
    parser = OptionParser(usage=usage)

    parser.add_option("-i", "--i", dest="input", nargs=1, default=None,
                      help="Enter a .gff or .bed file of binding sites used to make enhancers")
    parser.add_option("-r", "--rankby", dest="rankby", nargs=1, default=None,
                      help="bamfile to rank enhancer by")
    parser.add_option("-o", "--out", dest="out", nargs=1, default=None,
                      help="Enter an output folder")
    parser.add_option("-g", "--genome", dest="genome", default=None,
                      help="Enter the genome build (MM9,MM8,HG18,HG19,HG38)")
    parser.add_option("--custom", dest="custom_genome", default=None,
                      help="Enter the custom genome annotation refseq.ucsc")

    parser.add_option("-b", "--bams", dest="bams", nargs=1, default=None,
                      help="Enter a comma separated list of additional bam files to map to")
    parser.add_option("-c", "--control", dest="control", nargs=1, default=None,
                      help="bamfile to rank enhancer by")
    parser.add_option("-s", "--stitch", dest="stitch", nargs=1, default=12500,
                      help="Enter a max linking distance for stitching")
    parser.add_option("-t", "--tss", dest="tss", nargs=1, default=0,
                      help="Enter a distance from TSS to exclude. 0 = no TSS exclusion")
    parser.add_option("--no-gpu", dest="no_gpu", action='store_true', default=False,
                      help="Disable GPU acceleration, use CPU multiprocessing only.")

    (options, args) = parser.parse_args()

    if not options.input or not options.rankby or not options.out or not (options.genome or options.custom_genome):
        print('hi there')
        parser.print_help()
        exit()

    # ---- GPU System Summary ----
    use_gpu = not options.no_gpu
    if HAS_GPU:
        print_system_summary()
    else:
        print("GPU modules not available — using CPU mode")

    # ---- Setup folders ----
    outFolder = ROSE_utils.formatFolder(options.out, True)
    gffFolder = ROSE_utils.formatFolder(outFolder + 'gff/', True)
    mappedFolder = ROSE_utils.formatFolder(outFolder + 'mappedGFF/', True)

    # ---- Input file ----
    if options.input.split('.')[-1] == 'bed':
        inputGFFName = options.input.split('/')[-1][0:-4]
        inputGFFFile = '%s%s.gff' % (gffFolder, inputGFFName)
        ROSE_utils.bedToGFF(options.input, inputGFFFile)
    elif options.input.split('.')[-1] == 'gff':
        inputGFFFile = options.input
        os.system('cp %s %s' % (inputGFFFile, gffFolder))
    else:
        print('WARNING: INPUT FILE DOES NOT END IN .gff or .bed. ASSUMING .gff FILE FORMAT')
        inputGFFFile = options.input
        os.system('cp %s %s' % (inputGFFFile, gffFolder))

    # ---- BAM file list ----
    if options.control:
        bamFileList = [options.rankby, options.control]
    else:
        bamFileList = [options.rankby]

    if options.bams:
        bamFileList += options.bams.split(',')
        bamFileList = ROSE_utils.uniquify(bamFileList)

    # ---- Parameters ----
    stitchWindow = int(options.stitch)
    tssWindow = int(options.tss)
    removeTSS = tssWindow != 0

    print(('USING %s AS THE INPUT GFF' % (inputGFFFile)))
    inputName = inputGFFFile.split('/')[-1].split('.')[0]

    # ---- Annotation file ----
    cwd = os.getcwd()
    genomeDict = {
        'HG18': '%s/annotation/hg18_refseq.ucsc' % (cwd),
        'MM9': '%s/annotation/mm9_refseq.ucsc' % (cwd),
        'HG19': '%s/annotation/hg19_refseq.ucsc' % (cwd),
        'HG38': '%s/annotation/hg38_refseq.ucsc' % (cwd),
        'MM8': '%s/annotation/mm8_refseq.ucsc' % (cwd),
        'MM10': '%s/annotation/mm10_refseq.ucsc' % (cwd),
    }

    if options.custom_genome:
        annotFile = options.custom_genome
        print('USING CUSTOM GENOME %s AS THE GENOME FILE' % options.custom_genome)
    else:
        genome = options.genome
        annotFile = genomeDict[genome.upper()]
        print('USING %s AS THE GENOME' % genome)

    # ---- Start Dict ----
    print('MAKING START DICT')
    startDict = ROSE_utils.makeStartDict(annotFile)

    # ---- Load GFF regions ----
    print('LOADING IN GFF REGIONS')
    referenceCollection = ROSE_utils.gffToLocusCollection(inputGFFFile)

    # ---- Stitch regions ----
    print('STITCHING REGIONS TOGETHER')
    stitchedCollection, debugOutput = regionStitching(inputGFFFile, stitchWindow, tssWindow, annotFile, removeTSS)

    # ---- Write stitched GFF ----
    print('MAKING GFF FROM STITCHED COLLECTION')
    stitchedGFF = ROSE_utils.locusCollectionToGFF(stitchedCollection)

    if not removeTSS:
        stitchedGFFFile = '%s%s_%sKB_STITCHED.gff' % (gffFolder, inputName, stitchWindow / 1000)
        stitchedGFFName = '%s_%sKB_STITCHED' % (inputName, stitchWindow / 1000)
        debugOutFile = '%s%s_%sKB_STITCHED.debug' % (gffFolder, inputName, stitchWindow / 1000)
    else:
        stitchedGFFFile = '%s%s_%sKB_STITCHED_TSS_DISTAL.gff' % (gffFolder, inputName, stitchWindow / 1000)
        stitchedGFFName = '%s_%sKB_STITCHED_TSS_DISTAL' % (inputName, stitchWindow / 1000)
        debugOutFile = '%s%s_%sKB_STITCHED_TSS_DISTAL.debug' % (gffFolder, inputName, stitchWindow / 1000)

    if debug:
        print(('WRITING DEBUG OUTPUT TO DISK AS %s' % (debugOutFile)))
        ROSE_utils.unParseTable(debugOutput, debugOutFile, '\t')

    print(('WRITING STITCHED GFF TO DISK AS %s' % (stitchedGFFFile)))
    ROSE_utils.unParseTable(stitchedGFF, stitchedGFFFile, '\t')

    # ---- Setup output ----
    outputFile1 = outFolder + stitchedGFFName + '_REGION_MAP.txt'
    print(('OUTPUT WILL BE WRITTEN TO  %s' % (outputFile1)))

    nBin = 1
    rpm = True  # Enable RPM normalization

    # ================================================================
    # BAM MAPPING — Using concurrent Python calls instead of os.system
    # ================================================================
    print('\n' + '=' * 60)
    print('STARTING BAM DENSITY MAPPING')
    print('=' * 60)

    mapping_start = time.time()
    mapping_tasks = []

    for bamFile in bamFileList:
        bamFileName = bamFile.split('/')[-1]

        # Mapping to stitched GFF
        mappedOut1 = '%s%s_%s_MAPPED.gff' % (mappedFolder, stitchedGFFName, bamFileName)
        mapping_tasks.append((bamFile, stitchedGFFFile, mappedOut1))

        # Mapping to original GFF
        mappedOut2 = '%s%s_%s_MAPPED.gff' % (mappedFolder, inputName, bamFileName)
        mapping_tasks.append((bamFile, inputGFFFile, mappedOut2))

    # Execute mapping tasks concurrently using ThreadPoolExecutor
    # Each task handles its own GPU/CPU selection internally
    max_concurrent = 1  # Sequential to match original ROSE output ordering (critical for duplicate ID determinism)

    print(f'Running {len(mapping_tasks)} mapping tasks (max {max_concurrent} concurrent)')

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for bamFile, gffFile, mappedOut in mapping_tasks:
            future = executor.submit(
                run_bam_mapping,
                bamFile, gffFile, mappedOut,
                nBin, 200, True, 'both', use_gpu
            )
            futures[future] = mappedOut

        for future in as_completed(futures):
            output_file = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Mapping failed for {output_file}: {e}")
                raise

    mapping_elapsed = time.time() - mapping_start
    print(f'\n✅ ALL BAM MAPPING COMPLETED IN {mapping_elapsed:.1f} SECONDS')

    # ---- Map data to regions ----
    print('\nBAM MAPPING COMPLETED NOW MAPPING DATA TO REGIONS')
    mapCollection(stitchedCollection, referenceCollection, bamFileList, mappedFolder, outputFile1,
                  refName=stitchedGFFName)

    # ---- Call super-enhancers ----
    print('\nCALLING AND PLOTTING SUPER-STITCHED PEAKS')

    bin_dir = os.path.dirname(os.path.abspath(__file__))
    r_script = os.path.join(bin_dir, "ROSE_callSuper.R")
    py_script = os.path.join(bin_dir, "ROSE_geneMapper.py")

    if options.control:
        rankbyName = options.rankby.split('/')[-1]
        controlName = options.control.split('/')[-1]
        cmd = f"Rscript {r_script} {outFolder} {outputFile1} {inputName} {controlName}"
    else:
        rankbyName = options.rankby.split('/')[-1]
        controlName = 'NONE'
        cmd = f"Rscript {r_script} {outFolder} {outputFile1} {inputName} {controlName}"

    print(cmd)
    os.system(cmd)

    # ---- Gene mapping ----
    time.sleep(5)  # Brief pause for R script to finish writing
    superTableFile = "%s/%s_SuperStitched.table.txt" % (outFolder, inputName)
    allTableFile = "%s/%s_AllStitched.table.txt" % (outFolder, inputName)

    suffixScript = ''
    if options.control:
        suffixScript += '-c '
    if options.no_gpu:
        suffixScript += '--no-gpu '
        
    if options.custom_genome:
        cmd1 = f'"{sys.executable}" "{py_script}" --custom {options.custom_genome} -i {superTableFile} -r TRUE {suffixScript}'
        cmd2 = f'"{sys.executable}" "{py_script}" --custom {options.custom_genome} -i {allTableFile} -r TRUE {suffixScript}'
    else:
        cmd1 = f'"{sys.executable}" "{py_script}" -g {genome} -i {superTableFile} -r TRUE {suffixScript}'
        cmd2 = f'"{sys.executable}" "{py_script}" -g {genome} -i {allTableFile} -r TRUE {suffixScript}'

    print(cmd1)
    os.system(cmd1)

    print(cmd2)
    os.system(cmd2)

    print('\n🎉 ROSE PIPELINE COMPLETED SUCCESSFULLY')


if __name__ == "__main__":
    main()
