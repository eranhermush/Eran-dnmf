from gedit_preprocess.HandleInput import checkInputs
from gedit_preprocess.MatrixTools import RescaleRows, getSharedRows, remove0s, qNormMatrices
from gedit_preprocess.getSigGenesModal import returnSigMatrix


def run_gedit_pre1(rawMix, rawRef, use_all_genes=False, NumSigs=None, use_data=False):
    """
    usage default:
    python ThisScript.py -mix SamplesMat.tsv -ref RefMat.tsv

    user selected parameters:
    python ThisScript.py -mix SamplesMat.tsv -ref RefMat.tsv -numSigs SigsPerCT -method SigMethod -RS rowscaling
    """
    # where to write results
    curDir = "./"  # ""/".join(os.path.realpath(__file__).split("/")[0:-1]) + "/"
    # scratchSpace = curDir + "scratch/"

    if use_data:
        args_input = ["-mixData", rawMix, "-refData", rawRef]
    else:
        args_input = ["-mix", rawMix, "-ref", rawRef]
    if NumSigs is not None:
        args_input.extend(["-NumSigs", NumSigs])
    myArgs = checkInputs(args_input)

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

    stringParams = [str(m) for m in [MixFName, RefFName, SigsPerCT, SigMethod, RowScaling]]
    # refFile = scratchSpace + "signatures/" + "_".join(stringParams) + "_" + "ScaledRef.tsv"
    # mixFile = scratchSpace + "datasets/" + "_".join(stringParams) + "_" + "ScaledMix.tsv"
    # if os.path.exists(refFile) or os.path.exists(mixFile):
    #    return

    SampleNames = rawMix[0]
    CTNames = rawRef[0]

    betRef = remove0s(rawRef)
    # betRef = rawRef[rawRef.abs().sum(dim=1).bool(), :]

    normMix, normRef = qNormMatrices(rawMix, betRef)
    sharedMix, sharedRef = getSharedRows(normMix, betRef)

    if len(sharedMix) < 1 or len(sharedRef) < 1:
        print("error: no gene names match between reference and mixture")
        return
    if len(sharedMix) < numCTs or len(sharedRef) < numCTs:
        print("warning: only ", len(sharedMix), " gene names match between reference and mixture")

    # write normalized matrices
    # MatrixTools.writeMatrix([CTNames] + normRef, scratchSpace + "NormRef.tsv")
    # MatrixTools.writeMatrix([SampleNames] + normMix, scratchSpace + "NormMix.tsv")

    if not use_all_genes:
        SigRef = returnSigMatrix([CTNames] + sharedRef, SigsPerCT, TotalSigs, SigMethod)
    else:
        SigRef = sharedRef

    SigMix, SigRef = getSharedRows(sharedMix, SigRef)

    """
    write matrices with only sig genes. files are not used by this program,
    but potentially informative to the user
    """

    # eran:   MatrixTools.writeMatrix([CTNames] + SigRef, scratchSpace + "SigRef.tsv")
    # ScaledRef, ScaledMix = RescaleRows(SigRef[1:], SigMix[1:], RowScaling)
    ScaledRef, ScaledMix = RescaleRows(SigRef, SigMix, RowScaling)

    ScaledRef = [CTNames] + ScaledRef
    ScaledMix = [SampleNames] + ScaledMix

    # scratchSpace = scratchSpace + "_".join(stringParams) + "_"

    # MatrixTools.writeMatrix(ScaledRef, refFile)
    # MatrixTools.writeMatrix(ScaledMix, mixFile)
    return ScaledRef, ScaledMix
    # strDescr = "_".join(stringParams)
    # Rscript = "GLM_Decon.R"
    # if outFile == None:
    #     outFile = "predictions/" + "_".join(stringParams) + "Predictions.tsv"
    # print("Rscript", Rscript, mixFile, refFile, outFile)
    # predictions = Regression(scratchSpace, CTNames, SampleNames,refFile, mixFile,strDescr)
    # if predictions == False:
    #   return

    # for line in predictions:
    #  print "\t".join([str(el) for el in line])
    # return


def readInPredictions1(fname):
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


def gedit_main1(ref_file, mix_file, signature_name=None):
    REF_FOLDER = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/RefMats/"
    MIX_FOLDER = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/Mixes/"

    # ref_files = [f"{ref_folder}/{f}" for f in listdir(ref_folder) if isfile(join(ref_folder, f))]
    # if signature_name:
    #    ref_files = [f for f in ref_files if signature_name in f]
    # mix_files = [f"{mix_folder}/{f}" for f in listdir(mix_folder) if isfile(join(mix_folder, f))]

    # print("Working in ref=" + ref, " mix = " + mix)
    ScaledRef, ScaledMix = run_gedit_pre1(mix_file, ref_file)
    return ScaledRef, ScaledMix
