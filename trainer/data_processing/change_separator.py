input_csv = r'C:\Users\User\EasyOCR\trainer\all_data\en_sample\labels.csv'
output_csv = r'C:\Users\User\EasyOCR\trainer\all_data\en_sample\labels_fixed.csv'

with open(input_csv, 'r', encoding='utf-8') as infile, \
     open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
    
    for line in infile:
        # Replace only the first comma with tab
        # split only on first comma
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            new_line = parts[0] + '\t' + parts[1] + '\n'
        else:
            new_line = line  # For safety, if no comma found
        
        outfile.write(new_line)
