from numpy import *
import operator
import sys
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def get_cwd():
    return sys.path[0]

def BasicLearn():
    dataSet = array([[1.0, 1.1], [2.0, 2.1]])
    print("DataSet.Shape:" + str(dataSet.shape))
    tiled = tile(dataSet, [3, 1, 1])
    print("TiledResult:" + str(tiled))
    print("TiledDim:" + str(tiled.ndim))
    print("TiledShape:" + str(tiled.shape))
    print("DataSet**2:" + str(dataSet**2))


def CreateDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #compute distances
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print("DiffMat:" + str(diffMat))

    sqDiffMat = diffMat**2
    # print("SqDiffMat:" + str(sqDiffMat))

    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    # choose k most closed samples
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #sort
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []

    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix(
        get_cwd() + '/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0

    print "the total error rate is: %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(
        raw_input("percentage of time spent playing video game?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    icecream = float(raw_input("liters of icecream comsumed per year?"))
    # datingDataMat, datingLabels = file2matrix(
    #     get_cwd() + '/datingTestSet2.txt')
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, icecream])
    classifierResult = classify0(
        (inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]


#DigitRecognize

def img2vector(filename):
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32*i+j] = int(lineStr[j])

    return returnVector            

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(get_cwd() + "/trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(get_cwd() + "/trainingDigits/%s" % fileNameStr)
    
    testFileList = listdir(get_cwd() + '/testDigits')

    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(get_cwd() + '/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if(classifierResult != classNumStr): errorCount += 1.0

    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))



# img2vector(get_cwd() + '/testDigits/0_13.txt')
handwritingClassTest()

# group, labels = CreateDataSet()
# print(classify0([0, 0], group, labels, 3))


# datingDataMat, datingLabels = file2matrix(get_cwd() + '/datingTestSet2.txt')
# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)
# fig = plt.figure()
# ax = fig .add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#            15 * array(datingLabels), 15 * array(datingLabels))
# plt.show()

# datingClassTest()
# classifyPerson()
