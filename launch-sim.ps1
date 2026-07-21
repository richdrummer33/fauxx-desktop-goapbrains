$env:Path += ";$env:USERPROFILE\.cargo\bin"
cargo --version
$env:FAUXX_LLM = "1"
$env:FAUXX_LLM_ENDPOINT = "http://127.0.0.1:1234" 
# LAN Endpoint: http://169.254.83.107:1234
$env:FAUXX_LLM_MODEL = "phi-4-mini-3.8b-instruct"
$env:FAUXX_SIM_DAYS = "7"      # then 30 for the month
cargo run -p fauxx-core --example day_in_the_life
