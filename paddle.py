import os
import re
import json
import csv
from datetime import datetime
from paddleocr import PaddleOCR
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import traceback

# Constants
MIN_CONFIDENCE = 0.7
TIME_PATTERN = r'\d{2}:\d{2}:\d{2}'
BATCH_SIZE = 500
MAX_WORKERS = 6
MAX_RETRIES = 2
REST_EVERY = 20
REST_TIME = 20
CHUNK_SIZE = 20000  # Process in chunks of 20,000 images

def init_worker():
    global ocr
    try:
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            enable_mkldnn=True,
            cpu_threads=3
        )
    except Exception as e:
        print(f"Worker {os.getpid()} initialization failed: {str(e)}")
        raise

def parse_time_only(text):
    matches = re.findall(TIME_PATTERN, text)
    return matches[0] if matches else None

def process_image_batch(image_dir, directory_date, batch_files, batch_id):
    results = []
    init_worker()
    
    for idx, filename in enumerate(batch_files, 1):
        if idx % 100 == 0:
            print(f"[Batch {batch_id}] Processed {idx} images...")
        try:
            img_path = os.path.join(image_dir, filename)
            result = ocr.ocr(img_path, cls=True)
            
            extracted_texts = []
            confidences = []
            possible_timestamps = []
            digit_deficit = 14
            flags = []

            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    extracted_texts.append(text)
                    confidences.append(confidence)
                    
                    line_digits = sum(c.isdigit() for c in text)
                    digit_deficit = min(digit_deficit, max(14 - line_digits, 0))
                    
                    if time_part := parse_time_only(text):
                        full_timestamp = f"{directory_date} {time_part}"
                        possible_timestamps.append({
                            'timestamp': full_timestamp,
                            'confidence': confidence
                        })

            best_timestamp = None
            best_confidence = 0
            if possible_timestamps:
                possible_timestamps.sort(key=lambda x: x['confidence'], reverse=True)
                best_match = possible_timestamps[0]
                best_timestamp = best_match['timestamp']
                best_confidence = best_match['confidence']
                
                try:
                    datetime.strptime(best_timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    flags.append("invalid_format")

            if not possible_timestamps:
                flags.append("no_timestamp_found")
            elif best_confidence < MIN_CONFIDENCE:
                flags.append("low_confidence")

            results.append({
                'directory': os.path.basename(image_dir),
                'filename': filename,
                'parsed_timestamp': best_timestamp,
                'timestamp_confidence': best_confidence,
                'flags': flags,
                'digit_deficit': digit_deficit,
                'extracted_texts': extracted_texts,
                'confidences': confidences,
                'batch_id': batch_id,
                'batch_status': 'success'
            })
        except Exception as e:
            results.append({
                'directory': os.path.basename(image_dir),
                'filename': filename,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'batch_id': batch_id,
                'batch_status': 'failed'
            })
    
    return results

def process_directory(image_dir, max_workers=MAX_WORKERS, processed_files=None):
    folder_basename = os.path.basename(image_dir)
    match = re.search(r'\d{8}', folder_basename)
    if not match:
        raise ValueError(f"Cannot extract date from directory name: {folder_basename}")
    directory_date = datetime.strptime(match.group(), '%Y%m%d').strftime('%Y-%m-%d')

    processed_files = processed_files or set()

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg') and f not in processed_files],
        key=lambda x: (
            re.search(r'(\d{8}\d{6})', x).group(1),
            int(re.search(r'frame(\d+)\.jpg', x).group(1))
        )
    )[69000:]

    all_results = []
    total_processed = 0
    
    while len(processed_files) < len(image_files):
        remaining_files = [f for f in image_files if f not in processed_files]
        current_chunk = remaining_files[:CHUNK_SIZE]
        
        print(f"\nProcessing chunk of {len(current_chunk)} images (total processed: {len(processed_files)})")
        
        chunk_results = process_chunk(image_dir, directory_date, current_chunk, max_workers)
        all_results.extend(chunk_results)
        
        # Update processed files
        new_processed = {r['filename'] for r in chunk_results}
        processed_files.update(new_processed)
        total_processed += len(new_processed)
        
        # Save checkpoint after each chunk
        save_checkpoint(processed_files)
        print(f"Completed chunk. Total processed so far: {total_processed}")
        
        # Save intermediate results
        save_outputs(all_results, 'timestamps_output')
    
    return all_results

def process_chunk(image_dir, directory_date, chunk_files, max_workers):
    chunk_results = []
    batch_counter = 0
    batch_id = 0
    processed_images = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batches = [(i, chunk_files[i:i + BATCH_SIZE]) for i in range(0, len(chunk_files), BATCH_SIZE)]
        failed_batches = []
        initial_futures = []

        for batch_idx, batch in batches:
            batch_id += 1
            future = executor.submit(
                process_image_batch, 
                image_dir, 
                directory_date, 
                batch, 
                batch_id
            )
            future.batch_idx = batch_idx
            future.batch_id = batch_id
            initial_futures.append(future)
            batch_counter += 1
            
            processed_images += len(batch)
            if processed_images % 100 == 0:
                print(f"Processed {processed_images} images so far...")
            
            if batch_counter % REST_EVERY == 0:
                print(f"Processed {batch_counter} batches, resting...")
                time.sleep(REST_TIME)
        
        # Retry logic
        remaining_retries = MAX_RETRIES
        while remaining_retries >= 0:
            current_futures = initial_futures if remaining_retries == MAX_RETRIES else failed_batches.copy()
            failed_batches.clear()
            
            for future in as_completed(current_futures):
                try:
                    batch_results = future.result()
                    chunk_results.extend(batch_results)
                    print(f"Completed batch {future.batch_id} ({len(batch_results)} images)")
                except Exception as e:
                    print(f"Batch {future.batch_id} failed (attempt {MAX_RETRIES - remaining_retries + 1}/{MAX_RETRIES + 1}): {str(e)}")
                    if remaining_retries > 0:
                        new_future = executor.submit(
                            process_image_batch, 
                            image_dir, 
                            directory_date, 
                            batches[future.batch_idx][1], 
                            future.batch_id
                        )
                        new_future.batch_idx = future.batch_idx
                        new_future.batch_id = future.batch_id
                        failed_batches.append(new_future)
                    else:
                        chunk_results.extend({
                            'directory': os.path.basename(image_dir),
                            'filename': batches[future.batch_idx][1][i],
                            'error': f"Batch processing failed after {MAX_RETRIES + 1} attempts",
                            'batch_id': future.batch_id,
                            'batch_status': 'failed'
                        } for i in range(len(batches[future.batch_idx][1])))

            if not failed_batches:
                break
            remaining_retries -= 1
            if remaining_retries >= 0 and failed_batches:
                print(f"Retrying {len(failed_batches)} failed batches...")
                time.sleep(REST_TIME)

    return chunk_results

def save_outputs(data, base_name):
    with open(f'{base_name}.json', 'w') as f:
        json.dump({
            'metadata': {
                'total_images': len(data),
                'successful_batches': len(set(r['batch_id'] for r in data if r.get('batch_status') == 'success')),
                'failed_batches': len(set(r['batch_id'] for r in data if r.get('batch_status') == 'failed'))
            },
            'results': data
        }, f, indent=2)
    
    with open(f'{base_name}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Directory', 'Filename', 'Batch ID', 'Batch Status',
            'Parsed Timestamp', 'Timestamp Confidence', 'Flags', 
            'Digit Deficit', 'Extracted Texts', 'Error'
        ])
        for entry in data:
            writer.writerow([
                entry.get('directory', ''),
                entry.get('filename', ''),
                entry.get('batch_id', ''),
                entry.get('batch_status', ''),
                entry.get('parsed_timestamp', ''),
                f"{entry.get('timestamp_confidence', 0):.4f}",
                ', '.join(entry.get('flags', [])),
                entry.get('digit_deficit', ''),
                ' | '.join(entry.get('extracted_texts', [])),
                entry.get('error', '')
            ])

def load_checkpoint(file='processed_checkpoint.json'):
    if os.path.exists(file):
        with open(file, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(processed_set, file='processed_checkpoint.json'):
    with open(file, 'w') as f:
        json.dump(sorted(processed_set), f)

if __name__ == '__main__':
    CHECKPOINT_FILE = 'processed_checkpoint.json'
    
    if input("Start fresh run (clears checkpoint)? [y/N]: ").lower() == 'y':
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("Checkpoint cleared - starting fresh run")

    directories = [
        r'C:\Users\User\Documents\College\FYP2\20240110 - Copy'
    ]
    
    processed_set = load_checkpoint(CHECKPOINT_FILE)
    all_results = []
    start_time = time.time()

    for dir_path in directories:
        print(f"\nProcessing directory: {dir_path}")
        try:
            results = process_directory(
                dir_path,
                processed_files=processed_set
            )
            all_results.extend(results)
            print(f"Completed {len(results)} images from {dir_path}")

        except Exception as e:
            print(f"Error processing directory {dir_path}: {str(e)}")
            traceback.print_exc()

    end_time = time.time()

    save_outputs(all_results, 'timestamps_output')
    save_checkpoint(processed_set, CHECKPOINT_FILE)

    success_count = sum(1 for r in all_results if r.get('batch_status') == 'success')
    error_count = len(all_results) - success_count

    print("\nProcessing complete")
    print(f"Total images processed this run: {len(all_results)}")
    print(f"Successful batches: {len(set(r['batch_id'] for r in all_results if r.get('batch_status') == 'success'))}")
    print(f"Failed batches: {len(set(r['batch_id'] for r in all_results if r.get('batch_status') == 'failed'))}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

