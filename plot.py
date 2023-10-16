import matplotlib.pyplot as plt
import numpy as np

usingGoogleColab = True

if usingGoogleColab:
  from google.colab import files

iterationNames = []
timeCategory = []
times = []

currentTimeCategory = 0

with open("output.txt", "r") as file:
    for line in file:
      if ".out" in line:
        iterationNames.append(line.split(" ")[1].strip())
        currentTimeCategory = 0
      elif "microseconds" in line and "Writing to file" not in line and "Total time" not in line:
        if len(iterationNames) <= 1:
          times.append([])

        foundValues = line.split("microseconds")
        time = float(foundValues[0].strip()) / 1000
        title = foundValues[1].strip()

        times[currentTimeCategory].append(time)

        currentTimeCategory += 1

        if title not in timeCategory:
          timeCategory.append(title)

isFirst = True
currentBottom = None

for i in range(len(times)):
  currentTimes = np.array(times[i])

  if isFirst == True:
    plt.bar(iterationNames, currentTimes)
    isFirst = False
    currentBottom = currentTimes
  else:
    plt.bar(iterationNames, currentTimes, bottom=currentBottom)
    currentBottom += currentTimes

plt.xlabel("Iteration")
plt.ylabel("Execution Time [ms]")
plt.xticks(iterationNames, rotation=45, ha='right')
plt.title("Execution Times for Different Optimizations")
plt.legend(timeCategory)
plt.grid(True)
plt.savefig("plot.svg", format="svg")
plt.show()

if usingGoogleColab:
  files.download("plot.svg")
