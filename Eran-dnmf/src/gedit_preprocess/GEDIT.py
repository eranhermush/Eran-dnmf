import os
from os import listdir
from os.path import isfile, join

from gedit_preprocess import HandleInput, MatrixTools, getSigGenesModal


def run_gedit_pre(rawMix, rawRef, scratchSpace):
    """
    usage default:
    python ThisScript.py -mix SamplesMat.tsv -ref RefMat.tsv

    user selected parameters:
    python ThisScript.py -mix SamplesMat.tsv -ref RefMat.tsv -numSigs SigsPerCT -method SigMethod -RS rowscaling
    """
    # where to write results
    curDir = "./"  # ""/".join(os.path.realpath(__file__).split("/")[0:-1]) + "/"
    #scratchSpace = curDir + "scratch/"

    args_input = ["-mix", rawMix, "-ref", rawRef]
    myArgs = HandleInput.checkInputs(args_input)

    rawMix = myArgs[0]
    rawRef = myArgs[1]
    SigsPerCT = myArgs[2]
    # minSigs = myArgs[3]
    SigMethod = myArgs[4]
    RowScaling = myArgs[5]
    MixFName = myArgs[6].split("/")[-1]
    RefFName = myArgs[7].split("/")[-1]
    outFile = myArgs[8]

    numCTs = len(rawRef[0]) - 1
    TotalSigs = int(SigsPerCT * numCTs)

    stringParams = [str(m) for m in \
                    [MixFName, RefFName, SigsPerCT, SigMethod, RowScaling]]
    refFile = scratchSpace + "signatures/" + "_".join(stringParams) + "_" + "ScaledRef.tsv"
    mixFile = scratchSpace + "datasets/" + "_".join(stringParams) + "_" + "ScaledMix.tsv"
    if os.path.exists(refFile) or os.path.exists(mixFile):
        return

    SampleNames = rawMix[0]
    CTNames = rawRef[0]

    betRef = MatrixTools.remove0s(rawRef)

    normMix, normRef = MatrixTools.qNormMatrices(rawMix, betRef)
    sharedMix, sharedRef = MatrixTools.getSharedRows(normMix, betRef)

    if len(sharedMix) < 1 or len(sharedRef) < 1:
        print("error: no gene names match between reference and mixture")
        return
    if len(sharedMix) < numCTs or len(sharedRef) < numCTs:
        print("warning: only ", len(sharedMix), " gene names match between reference and mixture")

    # write normalized matrices
    # MatrixTools.writeMatrix([CTNames] + normRef, scratchSpace + "NormRef.tsv")
    # MatrixTools.writeMatrix([SampleNames] + normMix, scratchSpace + "NormMix.tsv")

    SigRef = getSigGenesModal.returnSigMatrix([CTNames] + sharedRef, \
                                              SigsPerCT, TotalSigs, SigMethod)

    SigMix, SigRef = MatrixTools.getSharedRows(sharedMix, SigRef)

    """
    write matrices with only sig genes. files are not used by this program,
    but potentially informative to the user
    """

    # eran:   MatrixTools.writeMatrix([CTNames] + SigRef, scratchSpace + "SigRef.tsv")
    ScaledRef, ScaledMix = MatrixTools.RescaleRows(SigRef[1:], SigMix[1:], RowScaling)

    ScaledRef = [CTNames] + ScaledRef
    ScaledMix = [SampleNames] + ScaledMix

    # scratchSpace = scratchSpace + "_".join(stringParams) + "_"

    MatrixTools.writeMatrix(ScaledRef, refFile)
    MatrixTools.writeMatrix(ScaledMix, mixFile)

    strDescr = "_".join(stringParams)
    Rscript = "GLM_Decon.R"
    if outFile == None:
        outFile = "predictions/" + "_".join(stringParams) + "Predictions.tsv"
    print("Rscript", Rscript, mixFile, refFile, outFile)
    # predictions = Regression(scratchSpace, CTNames, SampleNames,refFile, mixFile,strDescr)
    # if predictions == False:
    #   return

    # for line in predictions:
    #  print "\t".join([str(el) for el in line])
    return


def readInPredictions(fname):
    PredictionStream = open(fname, "r")
    predictions = []
    first = True
    for line in PredictionStream:
        parts = line.strip().split(",")
        if len(parts) < 2:  # i.e. its not csv, actually tsv
            parts = line.strip().split()
        if first:
            first = False
        else:
            parts = parts[1:]
        predictions.append(parts)
    return predictions


def gedit_main(ref_folder, mix_folder, output_folder, signature_name=None):
    REF_FOLDER = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/RefMats/"
    MIX_FOLDER = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/Mixes/"

    ref_files = [f"{ref_folder}/{f}" for f in listdir(ref_folder) if isfile(join(ref_folder, f))]
    if signature_name:
        ref_files = [f for f in ref_files if signature_name in f]
    mix_files = [f"{mix_folder}/{f}" for f in listdir(mix_folder) if isfile(join(mix_folder, f))]

    for ref in ref_files:
        for mix in mix_files:
            print("Working in ref=" + ref, " mix = " + mix)
            run_gedit_pre(mix, ref, output_folder)


if __name__ == '__main__':
    gedit_main()