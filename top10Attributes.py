import sys
import pprint
import time

from covidDeathModel import CovidDeathModel

def main():
    # Check cmdline args
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} trainingFile")
        print("Exiting...")
        sys.exit()

    # Parse cmdline args
    trainingFileName = sys.argv[1]

    # Start time
    startTime = time.time()

    model = CovidDeathModel(trainingFileName, 0)
    top10Attributes = model.getTopTenAttributes()

    # Print top 10 and time elapsed
    print(f"--- Top 10 attributes ---")
    pprint.pprint(top10Attributes)
    print(f"--- Time elapsed: {time.time() - startTime} seconds ---")

main()