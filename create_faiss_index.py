from create_vectors import extract_audio_features
import numpy as np
import faiss

# Create FAISS index (L2 distance)
d = 34  # Feature vector dimension
index = faiss.IndexFlatL2(d)

music_files = ["song1.mp3", "song2.mp3", "song3.mp3", "song4.mp3", "song5.mp3", "song6.mp3"]
song_vectors = []

for file in music_files:
    vec = extract_audio_features(file)
    song_vectors.append(vec)

# Convert to numpy array and add to FAISS
song_vectors = np.array(song_vectors).astype(np.float32)
index.add(song_vectors)

faiss.write_index(index, "music_db.index")

def find_similar_songs(query_file, index, song_vectors, top_k=5):
    query_vec = extract_audio_features(query_file).astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    similar_songs = [music_files[i] for i in indices[0]]
    
    return similar_songs

index = faiss.read_index("music_db.index")

song_path_to_find = input("Enter song path name: ")
similar = find_similar_songs(song_path_to_find, index, song_vectors, top_k=5)
print("Top 5 Similar Songs:", similar)
