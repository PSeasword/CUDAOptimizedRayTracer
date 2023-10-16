#!/bin/bash
counter=1
foundFiles=()
inputValues=()

echo "Found .out files:"

for foundFile in *.out; do
  displayName="$(basename $foundFile)"
  inputValues+=("$counter")

  echo "$counter. $displayName"
  ((counter++))
  foundFiles+=("$displayName")
done

if [[ "$counter" == 1 ]]; then
  echo "Could not find any .out files, use make first."
  exit 0
fi

echo "----------"
read -p "Enter number(s) to run (space as delimiter, leave empty for all): " inputFiles

if [[ "$inputFiles" != "" ]]; then
  IFS=" " read -ra inputValues <<< "$inputFiles"
fi

echo "----------"
echo "What to do:"
echo "1. Run"
echo "2. Average"
echo "3. nvprof"
echo "4. nsight"
echo "5. nvvp"

echo "----------"
read -p "Enter number for what to do: " inputOption

createImage=1

echo "----------"
read -p "Write to image file [Y/n]: " inputImage

if [[ "$inputImage" == "n" ]]; then
  createImage=0
fi

inputTotal=""
inputInformation=""

iterations=1

if [[ "$inputOption" != 2 ]] && [[ "$inputOption" != 4 ]]; then
  echo "----------"
  read -p "Show device information [Y/n]: " inputInformation
  
elif [[ "$inputOption" == 2 ]]; then
  echo "----------"
  read -p "Number of iterations (empty for 1): " inputAverage
  
  if [[ "$inputAverage" != "" ]]; then
    iterations="$inputAverage"
  fi
fi
  
echo "----------"

outputFile="output.txt"
> "$outputFile"

for value in "${inputValues[@]}"; do
  toRun="${foundFiles[value-1]}"

  echo "Running $toRun"

  echo "" >> "$outputFile"
  echo "" >> "$outputFile"
  echo "" >> "$outputFile"
  echo "==================== $toRun ====================" >> "$outputFile"

  if [[ "$inputOption" == 1 ]]; then
    if [[ "$inputInformation" != "n" ]]; then
      ./"$toRun" "$createImage" >> "$outputFile"
    else
      ./"$toRun" "$createImage" | tail -n +23 >> "$outputFile"
    fi
  
  elif [[ "$inputOption" == 2 ]]; then
    averages=(0 0 0 0 0 0 0 0 0)
    titles=()

    for ((i = 0; i < iterations; i++)); do
      echo "Iteration $i"

      titles=()

      totalTime=0
      currentIndex=0

      ./"$toRun" "$createImage" | tail -n +23 > tmp.txt

      while IFS= read -r line; do
        if [[ $line =~ ^[0-9] ]]; then
          foundNumber=$(echo "$line" | sed 's/[^0-9].*//')
          titles+=("$(echo "$line" | sed 's/[0-9]//g')")
          averages[currentIndex]=$((averages[currentIndex] + $foundNumber))

          if [[ "$currentIndex" != 6 ]]; then
            totalTime=$((totalTime + foundNumber))
          fi

          ((currentIndex++))
        fi
      done < tmp.txt

      averages[currentIndex]=$((averages[currentIndex] + totalTime))
      titles+=(" microseconds        Total time")
    done

    for ((j = 0; j < ${#averages[@]}; j++)); do
      average=$((averages[$j] / $iterations))
      echo "$average${titles[$j]}" >> "$outputFile"
    done
    
  elif [[ "$inputOption" == 3 ]]; then
    if [[ "$inputInformation" != "n" ]]; then
      nvprof ./"$toRun" "$createImage" >> "$outputFile"
    else
      nvprof ./"$toRun" "$createImage" | sed '1,22d' >> "$outputFile"
    fi

  elif [[ "$inputOption" == 4 ]]; then
    if [[ "$inputInformation" != "n" ]]; then
      /usr/local/cuda-11/bin/nv-nsight-cu-cli ./"$toRun" "$createImage" >> "$outputFile"
    else
      /usr/local/cuda-11/bin/nv-nsight-cu-cli ./"$toRun" "$createImage" | sed '3,23d' >> "$outputFile"
    fi

  elif [[ "$inputOption" == 5 ]]; then
    nvprof --output-profile "${toRun%.out}".nvvp -f ./"$toRun" "$createImage" >> "$outputFile"
  fi
done

cat "$outputFile"
