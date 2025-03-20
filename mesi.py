mesi= ("Gennaio  "+ #0 (1-1)*9
"Febbraio "+ #9 
"Marzo    "+ #18
"Aprile   "+ #27
"Maggio   "+ #36
"Giugno   "+ #45
"Luglio   "+ #54
"Agosto   "+ #63
"Settembre"+ #72
"Novembre "+ #81
"Dicembre ")  #90


numero_mese= int(input("Scrivi un numero da 1 a 12: "))
conversione= (numero_mese-1)*9
print(mesi[conversione:(conversione+9)])