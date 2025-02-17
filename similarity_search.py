from create_vectors import extract_audio_features
import faiss
def find_similar_songs(query_file, index, song_vectors, top_k=5):
    query_vec = extract_audio_features(query_file).astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    similar_songs = [music_files[i] for i in indices[0]]
    
    return similar_songs

index = faiss.read_index("music_db.index")

song_name_to_find = input("Enter filename: ")
similar = find_similar_songs(song_name_to_find, index, song_vectors, top_k=5)
print("Top 5 Similar Songs:", similar)

