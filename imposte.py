PRIMA_ALIQUOTA = 0.23
SECONDA_ALIQUOTA = 0.35
TERZA_ALIQUOTA = 0.43

reddito = float(input("Inserisci il tuo reddito: "))
tasse_da_scontare = 0.0
reddito_netto = 0.0

# Calcolo prima aliquota
if reddito > 28000:
    tasse_da_scontare += 28000 * PRIMA_ALIQUOTA
else:
    tasse_da_scontare += reddito * PRIMA_ALIQUOTA

# Calcolo seconda aliquota
if 28000 < reddito <= 50000:
    tasse_da_scontare += (reddito - 28000) * SECONDA_ALIQUOTA
elif reddito > 50000:
    tasse_da_scontare += 22000 * SECONDA_ALIQUOTA 

# Calcolo terza aliquota
if reddito > 50000:
    tasse_da_scontare += (reddito - 50000) * TERZA_ALIQUOTA

print("Le tasse da scontare sono:", tasse_da_scontare, "€")

reddito_netto = reddito - tasse_da_scontare
print("Il tuo reddito al netto delle tasse è:", reddito_netto, "€")
    
    


