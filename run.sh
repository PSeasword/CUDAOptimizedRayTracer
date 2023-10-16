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
echo "2. nvprof"
echo "3. nsight"
echo "4. nvvp"

echo "----------"
read -p "Enter number for what to do: " inputOption

inputTotal=""
inputInformation=""

if [[ "$inputOption" == 1 ]]; then
  echo "----------"
  read -p "Only total time (without write to file) [y/N]: " inputTotal
fi

if [[ "$inputTotal" != "y" ]] && [[ "$inputOption" != 4 ]]; then
  echo "----------"
  read -p "Show device information [Y/n]: " inputInformation
fi

echo "----------"

for value in "${inputValues[@]}"; do
  toRun="${foundFiles[value-1]}"

  echo ""
  echo ""
  echo ""
  echo "==================== $toRun ===================="

  if [[ "$inputOption" == 1 ]]; then
    if [[ "$inputTotal" == "y" ]]; then
      totalTime=0
      currentIndex=0

      ./"$toRun" > tmp.txt

      while IFS= read -r line; do
        if [[ $line =~ ^[0-9] ]]; then
          if [[ "$currentIndex" != 6 ]]; then
            totalTime=$((totalTime + $(echo "$line" | sed 's/[^0-9].*//')))
          fi

          ((currentIndex++))
        fi
      done < "tmp.txt"

      echo "Total time: $totalTime"
    elif [[ "$inputInformation" != "n" ]]; then
      ./"$toRun"
    else
      ./"$toRun" | tail -n +23
    fi
    
  elif [[ "$inputOption" == 2 ]]; then
    if [[ "$inputInformation" != "n" ]]; then
      nvprof ./"$toRun"
    else
      nvprof ./"$toRun" | sed '1,22d'
    fi

  elif [[ "$inputOption" == 3 ]]; then
    if [[ "$inputInformation" != "n" ]]; then
      /usr/local/cuda-11/bin/nv-nsight-cu-cli "$toRun"
    else
      /usr/local/cuda-11/bin/nv-nsight-cu-cli "$toRun" | sed '3,23d'
    fi

  elif [[ "$inputOption" == 4 ]]; then
    nvprof --output-profile "${toRun%.out}".nvvp -f ./"$toRun"
  fi
done
