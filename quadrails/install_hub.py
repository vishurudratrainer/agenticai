# Import 'install' from the 'hub' sub-module properly
from guardrails.hub import install

# Now call it
install("hub://guardrails/valid_length")
install("hub://guardrails/competitor_check")

print("Installation successful!")