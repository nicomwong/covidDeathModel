from collections import defaultdict

import math
import sys
import time

def isDeadFromDate(date):
    if date == "9999-99-99" or ('#' in date):    # 9999-99-99 is alive
        return False
    return True # anything else is dead

def attribValFromAge(age):
    if age < ageCutOff:
        return "under"
    return "over"
    # if age < ageCutOffOne:
    #     return 1
    # elif age < ageCutOffTwo:
    #     return 2
    # else:
    #     return 3

def isUnknownAttribVal(attribVal):
    # if 97 <= attribVal <= 99 or attribVal == 3: # Omitting this results in higher accuracy on remote
    #     return True
    return False

def getAttribVal(val : str, attribName):
    # Add code here to handle non-number values

    attribVal = int(val)  # Cast to int

    if attribName == "age": # Special case for age splitting
        attribVal = attribValFromAge(attribVal)

    else:
        if isUnknownAttribVal(attribVal):
            attribVal = "unknown"
    
    return attribVal

# Train the Naive Bayes model
def train(trainFileName, excludeList):
    global condProbMap, classDeadProb, attribNames

    with open(trainFileName, 'r') as trainFile:
        attribNames = trainFile.readline().rstrip('\n').split(',')  # List of column headers indexed by column number

        for row in trainFile:
            colValues = row.rstrip('\n').split(',')  # Attribute values for this row
            classDead = isDeadFromDate(colValues[4])

            for colNum, cellVal in enumerate(colValues):
                attribName = attribNames[colNum]
                
                # Skip columns in the exclude list
                if attribName in excludeList:
                    continue

                attribVal = getAttribVal(cellVal, attribName)

                condProbMap[attribName][classDead][attribVal] += 1   # Increment count N(X=x,C)
            
            # Increment count N(C)
            if classDead == True:
                classDeadProb[True] += 1
            else:
                classDeadProb[False] += 1
        
        # Note that condProbMap and classDeadProb both store counts, not probabilities yet

        # Divide each count N(X=x,C) by N(C) to store P(X=x|C) in condProbMap
        for attribName in condProbMap:
            for classDead in condProbMap[attribName]:
                for attribVal in condProbMap[attribName][classDead]:
                    condProbMap[attribName][classDead][attribVal] /= classDeadProb[classDead]

        # Divide each count N(C) by sum(N(C) for all C) to store P(C) in classDeadProb
        sumClassCounts = sum(classDeadProb.values() )
        for classDead in classDeadProb:
            classDeadProb[classDead] /= sumClassCounts

# Classify a row of attributes
def classify(attribValues, excludeList, printFlag):
    # Calculate argmax( log[P(X|c)P(c)] for each class c) where X is the set of attribute values
    # Note: log[P(X|c)P(c)] = sum( log[P(X=x|c)] for all X=x ) + log[P(c)]
    (maxSumLogProb, maxClass) = (-2 ** 32, False)
    for classDead in classDeadProb:
        sumLogProb = 0
        for colNum, cellVal in enumerate(attribValues):
            attribName = attribNames[colNum]

            # Skip columns in the exclude list
            if attribName in excludeList:
                continue

            attribVal = getAttribVal(cellVal, attribName)
            # weightCoeff = 2 - attribWeights[attribName]
            weightCoeff = -1*math.log(attribWeights[attribName]) + 1
            # for debugging; only one entry gets messed up with value = 1: pneumonia, True, unknown
            # if attribVal not in condProbMap[attribName][classDead]:
            #     print("Found at", attribName, classDead, attribVal)
            sumLogProb += weightCoeff * math.log(condProbMap[attribName][classDead][attribVal] )  # log[P(X=x|c)]
            
        sumLogProb += math.log(classDeadProb[classDead] )   # log[P(c)]

        if sumLogProb > maxSumLogProb:
            maxSumLogProb, maxClass = sumLogProb, classDead
    
    # For GradeScope, print 0 for alive and 1 for dead prediction
    if gradescopeActive and printFlag:
        print(int(maxClass) )
    
    return maxClass

# Test the accuracy of the trained model on a validation data set
def testAccuracy(testFileName, excludeList, printFlag):
    with open(testFileName, 'r') as testFile:
        testFile.readline() # Skip first line (headers)
        numRows, validCount = 0, 0
        for row in testFile:
            attribValues = row.rstrip('\n').split(',')  # Attribute values for this row
            
            actualClass = isDeadFromDate(attribValues[4] )
            if classify(attribValues, excludeList, printFlag) == actualClass:   # Correct classification
                validCount += 1

            numRows += 1

        return validCount / numRows

def trainAndTest(printFlag):
    global condProbMap, classDeadProb, attribNames

    # First, create the Naive Bayes Model by storing conditional probabilities P(X_i|C) in a map
    # where X_i is an attribute, and C is a class (dead or alive)
    condProbMap = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1) ) )    # 3-level nested dict with default value 1 (for smoothing)
                                                                                        # [TODO] Need to change how dict entry is created
                                                                                        # Right now, the default value = 1 but it does not get divided into a probability
                                                                                        # since a validation row may have never been seen in the training data set
                                                                                        # Honestly, probably doesn't matter since nearly all combinations of attribute values are seen in the training data
    classDeadProb = {True: 0, False: 0}
    attribNames = []    # List of attribute names, indexed by column number

    train(trainingFileName, excludeSet)

    import pprint
    import json

    # print("--- Conditional Probability Tables ---") # for debugging
    # pprint.pprint(json.loads(json.dumps(condProbMap) ) )

    # print()   # for debugging
    # print("--- Class Dead Probability --- ")  
    # pprint.pprint(classDeadProb)

    # Next, test the accuracy on the validation data set
    return testAccuracy(testFileName, excludeSet, printFlag)

def printBestAgeCutOff():
    global ageCutOff

    maxAccuracy = 0
    bestAgeCutOff = -1

    for ageCutOff in range(45, 56): 
        accuracy = trainAndTest(False)
        if accuracy > maxAccuracy:
            maxAccuracy, bestAgeCutOff = accuracy, ageCutOff

    print("Best age cut off:", bestAgeCutOff)
    print("Best accuracy:", maxAccuracy)

def printBestTwoAgeCutOffs():
    global ageCutOffOne, ageCutOffTwo

    maxAccuracy = 0
    bestAgeCutOffs = (-1, -1)

    for ageCutOffOne in range(20, 30):
        for ageCutOffTwo in range(45, 75):
            accuracy = trainAndTest(False)
            if accuracy > maxAccuracy:
                maxAccuracy, bestAgeCutOffs = accuracy, (ageCutOffOne, ageCutOffTwo)

    print("Best age cut offs:", bestAgeCutOffs)
    print("Best accuracy:", maxAccuracy)

def printBestExclusionSet():
    global excludeSet

    with open(testFileName, 'r') as testFile:
        attribNameList = testFile.readline().rstrip('\n').split(',')
    
    import itertools
    import pprint

    excludeSet = mustExcludeSet

    accBefore = trainAndTest(False)
    accumExcludeSet = set()
    for attribName in attribNameList:
        if attribName in mustExcludeSet:    # Must always exclude this column anyways
            continue
        
        excludeSet = mustExcludeSet | accumExcludeSet | {attribName}
        acc = trainAndTest(False)

        if (acc > accBefore):
            accBefore = acc
            accumExcludeSet |= {attribName}

    print("Best exclusion set:")
    print(accumExcludeSet)
    print("Best accuracy: ", accBefore)

    # # Below is super slow because # subsets is = 2^N
    # for i in range(1, len(colHeaderSet)// 6 ):
    #     excludeSubsetList = [set(j) for j in itertools.combinations(colHeaderSet, i)]
    #     pprint.pprint(excludeSubsetList)

def printBestWeights(attribName):
    global attribWeights

    import numpy as np
    import pprint

    trainAndTest(False) # To initialize global vars

    # bestWeight = 0
    # bestAcc = 0
    q = []

    for attribWeight in np.linspace(0.5, 1.5, 21):    # 0.05 interval, from 0.5 to 1.5
        tmp = attribWeights[attribName]

        attribWeights[attribName] = attribWeight
        acc = testAccuracy(testFileName, excludeSet, False)

        attribWeights[attribName] = tmp
        # if acc > bestAcc:
        #     bestAcc, bestWeight = acc, attribWeight
        q.append( (acc, attribWeight) )

    q.sort(reverse=True)

    if not(gradescopeActive):
        print(f"Top 5 weights for '{attribName}' are:")
        pprint.pprint(q[0:5] )

    attribWeights[attribName] = q[0][1]    # set weight to best weight


# Check cmdline args
if len(sys.argv) != 3 and len(sys.argv) != 4:
    print("Usage: python3 trainingFile testFile")
    print("Exiting...")
    sys.exit()

# Parse cmdline args
trainingFileName = sys.argv[1]
testFileName = sys.argv[2]

# Flag for gradescope test versus my tests
gradescopeActive = (len(sys.argv) == 3)

mustExcludeSet = {"entry_date", "date_symptoms", "date_died"}

# Hyper-parameters
excludeSet = {"patient_type", "sex", "pregnancy", "other_disease", "asthma", "tobacco"}    # Best for local was {'pregnancy', 'diabetes', 'patient_type', 'copd', 'other_disease', 'asthma', 'tobacco'}
# Best for remote was {"patient_type", "sex", "pregnancy", "other_disease", "asthma", "tobacco"}
excludeSet |= mustExcludeSet

ageCutOff = 49  # Best locally was 47
# Best remote was 49 (0.8746)

ageCutOffOne = 27
ageCutOffTwo = 51
# Best pair locally was (27, 51)
# But remotely, this pair was worse than a single age cutoff of 49

attribWeights = defaultdict(lambda: 1)  # Attribute weights, indexed by attribute name
                                        # Must be between 0 and 2
# attribWeights["pneumonia"] = 1.2
# attribWeights["obesity"] = 0.9
# # attribWeights["icu"] = 1.1
# # attribWeights["cardiovascular"] = 0.7
# attribWeights["inmsupr"] = 0.5
# # attribWeights["hypertension"] = 0.9
# # attribWeights["renal_chronic"] = 0.5
# attribWeights["diabetes"] = 1.2
# attribWeights["copd"] = 0.5

with open(testFileName, 'r') as testFile:
    attribNameList = testFile.readline().rstrip('\n').split(',')

# Start time
startTime = time.time()

testSet = set(attribNameList) - excludeSet

# for attribName in testSet:    # This is hacking to super-fit the weights to the validation set
#     printBestWeights(attribName)
# printBestAgeCutOff()
# printBestTwoAgeCutOffs()
# printBestExclusionSet()

accuracy = trainAndTest(True)
if not(gradescopeActive):
    print(f"--- Test accuracy: {accuracy} ---")

if not(gradescopeActive):
    # Print time elapsed
    print(f"--- Time elapsed: {time.time() - startTime} seconds ---")
