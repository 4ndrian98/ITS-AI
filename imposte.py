PRIMA_ALIQUOTA = 0.23
SECONDA_ALIQUOTA = 0.35
TERZA_ALIQUOTA = 0.43

SCAGLIONE_1 = 28000
SCAGLIONE_2 = 50000

reddito = float(input("Inserisci il tuo reddito: "))
tasse_da_scontare = 0.0
reddito_netto = 0.0

if reddito > SCAGLIONE_1:
    tasse_da_scontare += SCAGLIONE_1 * PRIMA_ALIQUOTA
else:
    tasse_da_scontare += reddito * PRIMA_ALIQUOTA

if reddito > SCAGLIONE_1:
    if reddito <= SCAGLIONE_2:
        tasse_da_scontare += (reddito - SCAGLIONE_1) * SECONDA_ALIQUOTA
    else:
        tasse_da_scontare += (SCAGLIONE_2 - SCAGLIONE_1) * SECONDA_ALIQUOTA

if reddito > SCAGLIONE_2:
    tasse_da_scontare += (reddito - SCAGLIONE_2) * TERZA_ALIQUOTA

print("Le tasse da pagare sono:", tasse_da_scontare, "€")

reddito_netto = reddito - tasse_da_scontare
print("Il tuo reddito al netto delle tasse è:", reddito_netto, "€")
