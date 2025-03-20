class Cliente:
    def __init__(self):
        self._totale_speso = 0
        self._totale_pagato = 0
        self._sconto_disponibile = False

    def effetuaAcquisto(self, ammontare):
        if self._sconto_disponibile:
            self._sconto_disponibile = False 
        self._totale_pagato += ammontare
        if self._totale_pagato >= 100:
            self._sconto_disponibile = True

    def scontoRaggiunto(self):
        if self._sconto_disponibile:
            self._sconto_disponibile = False
            return True
        else:
            return False
'''      
cliente= Cliente()
cliente.effetuaAcquisto(50) 
cliente.effetuaAcquisto(60)
print(cliente.scontoRaggiunto())

cliente.effetuaAcquisto(30)
'''
def main():
    cliente = Cliente()
    
    while True:
        try:
            # Chiedi all'utente l'importo dell'acquisto
            cliente_input = input("Inserisci l'importo dell'acquisto (o 'q' per uscire): ")
            if cliente_input.lower() == 'q':
                break
            ammontare = float(cliente_input)
            
            # Registra l'acquisto
            cliente.effetuaAcquisto(ammontare)
            
            # Mostra lo stato corrente
            print(f"\nTotale speso: {cliente.totale_pagato:.2f}€")
            if cliente.sconto_disponibile:
                print("Hai diritto a uno sconto di 10€ sul prossimo acquisto!")
            else:
                print("Nessuno sconto disponibile al momento.")
        except ValueError:
            print("Errore: Inserisci un numero valido.")

    print("\nProgramma terminato.")

if __name__ == "__main__":
    main()