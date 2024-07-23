# shellcheck disable=SC1113
#/usr/bin/zsh

# wolframscript -file analytical.wls rn=1 eps=10 maxOrder=2 numSamples=45
EPS=10
for sampleNumber in $(seq 26 8 26); do
  echo "Computing $sampleNumber..."
  wolframscript -file analytical_v2.wls  eps="$EPS" maxOrder=2 sampleNumber="$sampleNumber" parallel=False > \
    "results/output_eps_'$EPS'_maxOrder_2_rn_1_sampleNumber_'$sampleNumber'.txt" &
done
