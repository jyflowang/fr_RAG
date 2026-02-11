from tools.generate_embeddings import parse_file, generate_store_embeddings

def test_parse_file_metadata():
    test_filename = "GOOG_2025Q3.pdf"
    doc = parse_file(test_filename)
    
    # test if meta data is existed
    assert doc.metadata["file_name"] == test_filename
    
    # check if the slice is correct
    assert doc.metadata["company"] == "GOOG"
    assert doc.metadata["year"] == "2024"
    
    print("\n meta data is correct")
    

if __name__ == "__main__":
    test_parse_file_metadata()