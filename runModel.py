
import time
import sys

from covidDeathModel import CovidDeathModel

def main():
    # Check cmdline args
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(f"Usage: python3 {sys.argv[0]} trainingFile testFile")
        print("Exiting...")
        sys.exit()

    # Parse cmdline args
    trainingFileName = sys.argv[1]
    testFileName = sys.argv[2]

    # Flag for gradescope test versus my tests
    gradescopeActive = (len(sys.argv) == 3)

    # Start time
    startTime = time.time()

    # Run the training and testing
    model = CovidDeathModel(trainingFileName, gradescopeActive)
    model.train()   # Train
    accuracy = model.testAccuracy(testFileName) # Test

    # Print accuracy and time elapsed
    if not(gradescopeActive):
        print(f"--- Test accuracy: {accuracy} ---")
        print(f"--- Time elapsed: {time.time() - startTime} seconds ---")

main()