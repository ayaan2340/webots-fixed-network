import pickle

count = 0
while True:
    try:
        with open(f"bestBest/lebron{count}.pkl", "rb") as f:
            fitness_score, genome = pickle.load(f)
            print(fitness_score)
            count += 1
    except:
        break

f.close()