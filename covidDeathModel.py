
def isDeadFromDate(date):
    if date == "9999-99-99":    # 9999-99-99 is alive
        return False
    return True # anything else is dead

def attribValFromAge(age):
    if age < 50:
        return "under50"
    return "over50"

def isUnknownAttribVal(attribVal):
    if 97 <= attribVal <= 99:
        return True
    return False

# Train the Naive Bayes model
def train(trainFileName, excludeList):
    global condProbMap, classDeadProb

    with open(trainFileName, 'r') as trainFile:
        attribNames = trainFile.readline().rstrip('\n').split(',')  # List of column headers indexed by column number

        for rowNum, row in enumerate(trainFile):
            colValues = row.rstrip('\n').split(',')  # Attribute values for this row
            classDead = isDeadFromDate(colValues[4])

            for colNum, attribVal in enumerate(colValues):
                attribName = attribNames[colNum]
                
                # Skip columns in the exclude list
                if attribName in excludeList:
                    continue

                attribVal = int(attribVal)  # Cast to int

                if attribName == "age": # Special case for age splitting
                    attribVal = attribValFromAge(attribVal)

                else:
                    if isUnknownAttribVal(attribVal):
                        attribVal = "unknown"

                # Count N(X=x,C)
                if attribName not in condProbMap:
                    condProbMap[attribName] = dict()    # Init. level 1 of dict
                if classDead not in condProbMap[attribName]:
                    condProbMap[attribName][classDead] = dict()  # Init. level 2
                if attribVal not in condProbMap[attribName][classDead]:
                    condProbMap[attribName][classDead][attribVal] = 0    # Init. level 3

                condProbMap[attribName][classDead][attribVal] += 1   # Increment count
            
            # Count N(C)
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

# Test the accuracy of the trained model
def testAccuracy(testFileName):
    pass

# First, create the Naive Bayes Model by storing conditional probabilities P(X_i|C) in a map
# where X_i is an attribute, and C is a class (dead or alive)
condProbMap = {}
excludeList = ["entry_date", "date_symptoms", "date_died"]
classDeadProb = {True: 0, False: 0}

train("covid_train.csv", excludeList)

import pprint
print("--- Conditional Probability Tables ---")
pprint.pprint(condProbMap)

print()
print("--- Class Dead Probability --- ")
pprint.pprint(classDeadProb)

# Next, test the accuracy on the validation data set
print(testAccuracy("covid_valid.csv") )