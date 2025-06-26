for compressed_file in ../data/vascular_data_jvs/*.zip; do
  # Unzip the compressed file into a temporary folder
  unzip "$compressed_file" -d temp_unzipped_folder
  # Iterate through all PDF files in the unzipped folder
  for pdf_file in temp_unzipped_folder/*.pdf; do
    # Extract text from the PDF file and store it in a variable
    extracted_text=$(pdftotext "$pdf_file" -)
    # Make the whole article be one line by replacing newlines with spaces
    processed_text=$(echo "$extracted_text" | tr '\n' ' ')
    # Append the processed text to the output file
    echo "$processed_text" >> ../data/vascular_data_jvs.txt
  done
  # Clean up the temporary folder
  rm -rf temp_unzipped_folder
done