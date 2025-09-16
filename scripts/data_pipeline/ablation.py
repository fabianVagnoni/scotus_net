import os

def ablation(justice_bio:str, to_ablate:set, justice_bio_ablated:str):
    with open(justice_bio, "r") as d:
        bio = d.read()
    bio = bio.splitlines()
    # Filter out lines that start with any of the ablation terms
    filtered_bio = []
    for line in bio:
        if line.strip():  # Skip empty lines
            should_keep = True
            for term in to_ablate:
                if line.startswith(term + ":"):
                    should_keep = False
                    break
            if should_keep:
                filtered_bio.append(line)
    
    bio = "\n".join(filtered_bio)
    with open(justice_bio_ablated, "w") as d:
        d.write(bio)

def main():
    justice_bio_dir = "data/processed/bios"
    justice_bio_ablated_dir = "data/processed/ablated_bios"
    to_ablate = {"Justice","State","Position","Appointment","Nominated","Birth",
    "Birth Year", "Appointment Date", "Nominated By"}
    if not os.path.exists(justice_bio_ablated_dir):
        os.makedirs(justice_bio_ablated_dir)

    for justice_bio in os.listdir(justice_bio_dir)[:10]:
        print(f"Ablating {justice_bio}")
        justice_bio_ablated = os.path.join(justice_bio_ablated_dir, justice_bio.replace(".txt", "_ablated.txt"))
        justice_bio = os.path.join(justice_bio_dir, justice_bio)
        ablation(justice_bio, to_ablate, justice_bio_ablated)

if __name__ == "__main__":
    main()