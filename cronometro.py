import time 
tempo = int(input("Scrivi in secondi il tempo: "))

for x in range(tempo, 0, -1):
    secondi= x%60
    minuti= int(x/60)%60
    ore=int(x/3600)
    time.sleep(1)
    print(f"{ore:02}:{minuti:02}:{secondi:02}")

print("Tempo Finito!!!")
