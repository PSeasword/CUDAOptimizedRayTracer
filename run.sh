counter=1
foundFiles=()

echo "Found .out files:"

for foundFile in *.out; do
  displayName="$(basename $foundFile)"

  echo "$counter. $displayName"
  ((counter++))
  foundFiles+=("$displayName")
done

if [[ "$counter" == 1 ]]; then
  echo "Could not find any .out files, use make first."
  exit 0
fi

echo "----------"
read -p "Enter number(s) to run (space as delimiter): " inputFiles
IFS=" " read -ra inputValues <<< "$inputFiles"

echo "----------"
echo "What to do:"
echo "1. Run"
echo "2. nvprof"
echo "3. nsight"
echo "4. nvvp"

echo "----------"
read -p "Enter number for what to do: " inputOption
echo "----------"

for value in "${inputValues[@]}"; do
  toRun="${foundFiles[value-1]}"

  if [[ "$inputOption" == 1 ]]; then
    ./"$toRun"
  elif [[ "$inputOption" == 2 ]]; then
    nvprof ./"$toRun"
  elif [[ "$inputOption" == 3 ]]; then
    /usr/local/cuda-11/bin/nv-nsight-cu-cli "$toRun"
  elif [[ "$inputOption" == 4 ]]; then
    nvprof --output-profile "${toRun%.out}".nvvp -f ./"$toRun"
  fi
done
