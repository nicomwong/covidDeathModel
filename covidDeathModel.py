from collections import defaultdict

import math
import time
import sys

# This Naive Bayes Classifier Model can be trained from a training file
# and then used on a test/validation data set to get accuracy.
# The formatting of these files is strict and must follow covid_train.csv
class CovidDeathModel:

    def __init__(self, trainFileName, gradescopeActive):
        # First, create the Naive Bayes Model by storing conditional probabilities P(X_i|C) in a map
        # where X_i is an attribute, and C is a class (dead or alive)
        self.condProbMap = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1) ) )    # 3-level nested dict with default value 1
                                                                                            # [TODO] Need to change how dict entry is created
                                                                                            # Right now, the default value = 1 but it does not get divided into a probability
                                                                                            # since a validation row may have never been seen in the training data set
                                                                                            # Honestly, probably doesn't matter since nearly all combinations of attribute values are seen in the training data
        self.classDeadProb = {True: 0, False: 0}    # Stores P(C) where C is a class (T: dead, F: alive)
        self.trainFileName = trainFileName

        # Get the list of attribute names, indexed by column number
        with open(trainFileName, 'r') as trainFile:
            self.attribNames = trainFile.readline().rstrip('\n').split(',')
        
        # Hyper-parameters
        self.excludeSet = {"entry_date", "date_symptoms", "date_died"} | {"patient_type", "sex", "pregnancy", "other_disease", "asthma", "tobacco"}
        self.ageCutOff = 49

        # Testing flags
        self.gradescopeActive = gradescopeActive

    # Train the Naive Bayes model
    def train(self):

        with open(self.trainFileName, 'r') as trainFile:

            trainFile.readline()    # Skip first line (headers)

            for row in trainFile:
                colValues = row.rstrip('\n').split(',')  # Attribute values for this row
                classDead = self.isDeadFromDate(colValues[4])

                for colNum, cellVal in enumerate(colValues):
                    attribName = self.attribNames[colNum]
                    
                    # Skip columns in the exclude list
                    if attribName in self.excludeSet:
                        continue

                    attribVal = self.getAttribVal(cellVal, attribName)

                    self.condProbMap[attribName][classDead][attribVal] += 1   # Increment count N(X=x,C)
                
                # Increment count N(C)
                self.classDeadProb[classDead] += 1

            # Note that at this point, condProbMap and classDeadProb both store counts, not probabilities yet

            # Divide each count N(X=x,C) by N(C) to store P(X=x|C) in condProbMap
            for attribName in self.condProbMap:
                for classDead in self.condProbMap[attribName]:
                    for attribVal in self.condProbMap[attribName][classDead]:
                        self.condProbMap[attribName][classDead][attribVal] /= self.classDeadProb[classDead]

            # Divide each count N(C) by sum(N(C) for all C) to store P(C) in classDeadProb
            sumClassCounts = sum(self.classDeadProb.values() )
            for classDead in self.classDeadProb:
                self.classDeadProb[classDead] /= sumClassCounts

    # Test the accuracy of the trained model on a validation data set
    def testAccuracy(self, testFileName):
        with open(testFileName, 'r') as testFile:
            testFile.readline() # Skip first line (headers)

            numRows, validCount = 0, 0
            # Calculate accuracy
            for row in testFile:
                attribValues = row.rstrip('\n').split(',')  # Attribute values for this row
                
                actualClass = self.isDeadFromDate(attribValues[4] )
                if self.classify(attribValues) == actualClass:
                    # Correct classification
                    validCount += 1

                numRows += 1

            return validCount / numRows

    # Classify a row of attributes
    def classify(self, attribValues):
        # Calculate argmax( log[P(X|c)P(c)] for each class c) where X is the set of attribute values
        # Note: log[P(X|c)P(c)] = sum( log[P(X=x|c)] for all X=x ) + log[P(c)]

        (maxSumLogProb, maxClass) = (-2 ** 32, False)

        for classDead in self.classDeadProb:
            sumLogProb = 0
            for colNum, cellVal in enumerate(attribValues):
                attribName = self.attribNames[colNum]

                # Skip columns in the exclude list
                if attribName in self.excludeSet:
                    continue

                attribVal = self.getAttribVal(cellVal, attribName)

                # for debugging; only one entry gets messed up with value = 1: pneumonia, True, unknown
                # if attribVal not in self.condProbMap[attribName][classDead]:
                #     print("Found at", attribName, classDead, attribVal)

                sumLogProb += math.log(self.condProbMap[attribName][classDead][attribVal] )  # add log[P(X=x|c)]
                
            sumLogProb += math.log(self.classDeadProb[classDead] )   # add log[P(c)]

            if sumLogProb > maxSumLogProb:
                maxSumLogProb, maxClass = sumLogProb, classDead
        
        # For GradeScope, print 0 for alive and 1 for dead prediction
        if self.gradescopeActive:
            print(int(maxClass) )
        
        return maxClass

    def isDeadFromDate(self, date):
        if date == "9999-99-99" or ('#' in date):    # 9999-99-99 is alive
            return False
        return True # anything else is dead

    def getAttribVal(self, val : str, attribName):
        # Add code here to handle non-number values

        attribVal = int(val)  # Cast to int

        if attribName == "age": # Special case for age splitting
            attribVal = self.attribValFromAge(attribVal)

        else:
            if self.isUnknownAttribVal(attribVal):
                attribVal = "unknown"
        
        return attribVal
    
    def attribValFromAge(self, age):
        if age < self.ageCutOff:
            return "under"
        return "over"

    def isUnknownAttribVal(self, attribVal):
        if 97 <= attribVal <= 99 or attribVal == 3: # Omitting this results in higher accuracy on remote
            return True
        return False

    def getTopTenAttributes(self):

        condProbMap = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1) ) )

        with open(self.trainFileName, 'r') as trainFile:

            trainFile.readline()    # Skip first line (headers)

            for row in trainFile:
                colValues = row.rstrip('\n').split(',')  # Attribute values for this row
                classDead = self.isDeadFromDate(colValues[4])

                for colNum, cellVal in enumerate(colValues):
                    attribName = self.attribNames[colNum]
                    
                    # Skip columns in the exclude list
                    if attribName in self.excludeSet:
                        continue

                    attribVal = self.getAttribVal(cellVal, attribName)

                    condProbMap[attribName][classDead][attribVal] += 1   # Increment count N(X=x,C)
                
                # Increment count N(C)
                self.classDeadProb[classDead] += 1

        # [TODO]      
        # Get total of N(X=x) for each x

    # Calculates I(x,y) = -x/(x+y)*log(x/(x+y)) - y/(x+y)*log(y/(x+y))
    def calcEntropy(x, y):
        total = x + y
        return -1 * (x / total * math.log(x / total, 2) + y / total * math.log(y / total, 2) )