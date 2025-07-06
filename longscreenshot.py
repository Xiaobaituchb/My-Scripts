#!/usr/bin/env python3
import cv2
import numpy as np
import subprocess
import sys
import time
import math
import signal
import os
from pathlib import Path
import traceback
from collections import Counter
import obsws_python as obs
from PIL import Image
import io
import atexit
from multiprocessing import Process, Queue


SIFT_MATCH_RATIO = 0.4  # Balanced ratio
MIN_MATCH_COUNT = 50    # Reasonable minimum
DEBUG_MATCHES = False
OBS_SOURCE_NAME = "Screen Capture (PipeWire)"
JUMP = 8
FACTOR=0.5
FLY= 64
MAX_GOOD_MATCHES = 3000
MAX_FEATURES = 5000  # Add this line
LOCK_FILE = Path("/tmp/longscreenshot.lock")


class Stitcher:
    def __init__(self, match_ratio=SIFT_MATCH_RATIO, min_matches=MIN_MATCH_COUNT, debug=DEBUG_MATCHES):
        self.images = []
        self.copy_chain = []
        self.last_keypoints = None
        self.last_descriptors = None
        self.debug = debug
        self.min_matches = min_matches

        try:
            # Modify this line to limit the number of features
            self.sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
        except cv2.error as e:
            print(f"Error creating SIFT: {e}")
            print("Ensure 'opencv-contrib-python' is installed (e.g., pip install opencv-contrib-python).")
            exit(1)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.match_ratio = match_ratio

    def add_image(self, image):
        current_image = image
        if current_image is None or current_image.size == 0:
            print("MATCH FAILURE: Received empty or null image")
            return False

        try:
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(f"MATCH FAILURE: Error converting image to grayscale - {e}")
            return False

        new_keypoints, new_descriptors = self.sift.detectAndCompute(gray, None)

        if new_keypoints is None or new_descriptors is None:
            print("MATCH FAILURE: SIFT failed to detect any keypoints or descriptors in current image")
            if not self.images:
                self.images.append(current_image.copy())
                self.last_keypoints = new_keypoints
                self.last_descriptors = new_descriptors
                return True
            else:
                return True
        
        if len(new_keypoints) < self.min_matches:
            print(f"MATCH FAILURE: Too few keypoints detected ({len(new_keypoints)} < {self.min_matches} required)")
            if not self.images:
                self.images.append(current_image.copy())
                self.last_keypoints = new_keypoints
                self.last_descriptors = new_descriptors
                return True
            else:
                return True

        if not self.images:
            self.last_keypoints = new_keypoints
            self.last_descriptors = new_descriptors
            self.images.append(current_image.copy())
            print("First image added successfully.")
            return True

        if self.last_descriptors is None or self.last_keypoints is None:
            print("MATCH FAILURE: Previous image has no valid keypoints/descriptors")
            self.images.append(current_image.copy())
            self.copy_chain.append((0, 0))
            self.last_keypoints = new_keypoints
            self.last_descriptors = new_descriptors
            return True
            
        if len(self.last_keypoints) < self.min_matches:
            print(f"MATCH FAILURE: Previous image has too few keypoints ({len(self.last_keypoints)} < {self.min_matches} required)")
            self.images.append(current_image.copy())
            self.copy_chain.append((0, 0))
            self.last_keypoints = new_keypoints
            self.last_descriptors = new_descriptors
            return True

        try:
            if self.last_descriptors.dtype != np.float32: self.last_descriptors = self.last_descriptors.astype(np.float32)
            if new_descriptors.dtype != np.float32: new_descriptors = new_descriptors.astype(np.float32)

            if self.last_descriptors.shape[0] == 0 or new_descriptors.shape[0] == 0:
                print("MATCH FAILURE: One of the descriptor arrays is empty")
                return True

            if self.last_descriptors.shape[1] != new_descriptors.shape[1]:
                print(f"MATCH FAILURE: Descriptor dimension mismatch! Previous={self.last_descriptors.shape[1]}, Current={new_descriptors.shape[1]}")
                return False

            matches = self.matcher.knnMatch(self.last_descriptors, new_descriptors, k=2)

        except Exception as e:
            print(f"MATCH FAILURE: Error during KNN matching - {e}")
            traceback.print_exc()
            return True

        if not matches:
            print("MATCH FAILURE: KNN matcher returned no matches")
            return True

        good_matches = []
        single_matches = 0
        filtered_by_ratio = 0
        
        for match_pair in matches:
            if len(match_pair) == 1:
                single_matches += 1
                continue
            elif len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
                else:
                    filtered_by_ratio += 1

        print(f"MATCH INFO: Raw matches={len(matches)}, Single matches={single_matches}, Filtered by ratio test={filtered_by_ratio}, Good matches={len(good_matches)}")

        if len(good_matches) > MAX_GOOD_MATCHES:
            print(f"INFO: Exceeded match limit. Pruning from {len(good_matches)} to the best {MAX_GOOD_MATCHES}.")
            good_matches.sort(key=lambda x: x.distance)  # Sort by distance (lower is better)
            good_matches = good_matches[:MAX_GOOD_MATCHES]
        # Require good matches but not too strict
        if len(good_matches) < max(round(len(matches)*0.05),self.min_matches):
            print(f"MATCH FAILURE: Insufficient good matches ({len(good_matches)} < {max(round(len(matches)*0.05),self.min_matches)} required)")
            print(f"  - Consider: Images may be too different, poor lighting, or excessive blur")
            print(f"  - Raw matches found: {len(matches)}")
            print(f"  - Ratio test eliminated: {filtered_by_ratio} matches")
            return True

        # Extract matched point coordinates
        src_pts = np.float32([self.last_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([new_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # ... inside the add_image method, after defining src_pts and dst_pts ...

        try:
            # Define RANSAC parameters
            ransac_reproj_threshold = 0.3
            ransac_confidence = 0.99
            ransac_max_iters = 2000

            print(f"INFO: Running RANSAC with up to {ransac_max_iters} iterations...")

            # --- NEW CODE: Start the timer ---
            start_time = time.perf_counter()

            # Use reasonable RANSAC parameters
            homography, mask = cv2.findHomography(
                dst_pts, src_pts,
                cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_threshold,
                confidence=ransac_confidence,
                maxIters=ransac_max_iters
            )

            # --- NEW CODE: Stop the timer and calculate duration ---
            end_time = time.perf_counter()
            ransac_duration_ms = (end_time - start_time) * 1000

            if homography is None:
                print("MATCH FAILURE: findHomography returned None - points may be coplanar or degenerate")
                return True

            if mask is None:
                print("MATCH FAILURE: findHomography mask is None - unexpected error")
                return True

            inliers = np.sum(mask)
            inlier_ratio = inliers / len(good_matches)

            # --- REVISED: Report the duration along with the results ---
            print(f"INFO: RANSAC finished in {ransac_duration_ms:.2f} ms. Found {inliers} inliers out of {len(good_matches)} matches.")

            # Reasonable requirements for accepting a match
            MIN_INLIERS = len(good_matches) * 0.3

            if inliers < MIN_INLIERS:
                print(f"MATCH FAILURE: Too few inliers ({inliers} < {MIN_INLIERS:.1f} required)")
                print(f"  - Inlier ratio: {inlier_ratio:.2f}")
                print(f"  - This suggests the images don't have consistent geometric relationship")
                print(f"  - Possible causes: Different content, perspective change, or excessive noise")
                return True

            # Extract homography components for validation
            h = homography

            # Extract translation components
            dx = int(round(h[0, 2]))
            dy = int(round(h[1, 2]))

            displacement = (dx, dy)

        except cv2.error as e:
            print(f"MATCH FAILURE: OpenCV error in homography estimation - {e}")
            return True
        except Exception as e:
            print(f"MATCH FAILURE: Unexpected error in displacement calculation - {e}")
            traceback.print_exc()
            return True

# ... the rest of the method continues ...

        # Rest of the method remains the same...
        if self.debug:
            # Debug visualization code (unchanged)
            prev_img_display = self.images[-1]
            if len(prev_img_display.shape) == 2 or prev_img_display.shape[2] == 1:
                prev_img_display = cv2.cvtColor(prev_img_display, cv2.COLOR_GRAY2BGR)
            elif prev_img_display.shape[2] == 4:
                prev_img_display = cv2.cvtColor(prev_img_display, cv2.COLOR_BGRA2BGR)

            curr_img_display = current_image
            if len(curr_img_display.shape) == 2 or curr_img_display.shape[2] == 1:
                curr_img_display = cv2.cvtColor(curr_img_display, cv2.COLOR_GRAY2BGR)
            elif curr_img_display.shape[2] == 4:
                curr_img_display = cv2.cvtColor(curr_img_display, cv2.COLOR_BGRA2BGR)

            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
            img_matches = cv2.drawMatches(prev_img_display, self.last_keypoints, curr_img_display, new_keypoints, inlier_matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            try:
                cv2.imshow('Matches', cv2.resize(img_matches, None, fx=0.5, fy=0.5))
                cv2.waitKey(1)
            except cv2.error as e:
                if "display" in str(e).lower() or "GTK" in str(e).upper() or "cannot open display" in str(e).lower():
                    print("Debug window disabled (No display available, permissions issue, or Wayland restriction)")
                    self.debug = False
                else:
                    print(f"cv2.imshow error in debug: {e}")

        self.copy_chain.append(displacement)
        self.images.append(current_image.copy())
        self.last_keypoints = new_keypoints
        self.last_descriptors = new_descriptors
        print(f"MATCH SUCCESS: Added image {len(self.images)} with shift {displacement} (inliers: {inliers}/{len(good_matches)}, ratio: {inlier_ratio:.2f})")
        global JUMP
        JUMP=round(max(1,min(FLY,JUMP+10*(1-FACTOR/inlier_ratio))))
        return True

    def stitch(self):
        if not self.images:
            print("No images to stitch.")
            return None
        if len(self.images) < 2:
            print("Not enough images with detected movement to stitch. Returning first image.")
            return self.images[0]

        print("Calculating final canvas...")
        positions = [(0, 0)]
        current_x, current_y = 0, 0
        if len(self.images) != len(self.copy_chain) + 1:
             print(f"Warning: Mismatch image count ({len(self.images)}) and copy chain ({len(self.copy_chain)}). Stitching might be inaccurate.")

        for i, (dx, dy) in enumerate(self.copy_chain):
            current_x += dx
            current_y += dy
            positions.append((current_x, current_y))

        while len(positions) < len(self.images):
             print(f"Warning: Adding default (last known) position for image {len(positions)}")
             positions.append(positions[-1])

        min_x, min_y = float('inf'), float('inf')
        max_x_coord, max_y_coord = float('-inf'), float('-inf')

        for i, (x_pos, y_pos) in enumerate(positions):
             if i >= len(self.images): break
             h, w = self.images[i].shape[:2]
             min_x = min(min_x, x_pos)
             min_y = min(min_y, y_pos)
             max_x_coord = max(max_x_coord, x_pos + w)
             max_y_coord = max(max_y_coord, y_pos + h)

        if min_x == float('inf'):
            print("Error: Could not determine image positions for stitching.")
            return self.images[0]

        canvas_width = max_x_coord - min_x
        canvas_height = max_y_coord - min_y
        offset_x = -min_x
        offset_y = -min_y

        if canvas_width <= 0 or canvas_height <= 0:
             print(f"Error: Calculated canvas size is invalid ({int(canvas_width)}x{int(canvas_height)}). Returning first image.")
             return self.images[0]

        print(f"Canvas size: {int(canvas_width)} x {int(canvas_height)}")
        result = np.zeros((int(canvas_height), int(canvas_width), 3), dtype=np.uint8)

        print("Pasting images...")
        for i, img in enumerate(self.images):
             if i >= len(positions): break
             x_pos, y_pos = positions[i]
             paste_x = int(x_pos + offset_x)
             paste_y = int(y_pos + offset_y)
             h, w = img.shape[:2]

             y_start_res, y_end_res = paste_y, paste_y + h
             x_start_res, x_end_res = paste_x, paste_x + w
             y_start_img, y_end_img = 0, h
             x_start_img, x_end_img = 0, w

             if y_start_res < 0: y_start_img = -y_start_res; y_start_res = 0
             if x_start_res < 0: x_start_img = -x_start_res; x_start_res = 0
             if y_end_res > result.shape[0]: y_end_img -= (y_end_res - result.shape[0]); y_end_res = result.shape[0]
             if x_end_res > result.shape[1]: x_end_img -= (x_end_res - result.shape[1]); x_end_res = result.shape[1]

             if y_end_res > y_start_res and x_end_res > x_start_res:
                 try:
                    result[y_start_res:y_end_res, x_start_res:x_end_res] = img[y_start_img:y_end_img, x_start_img:x_end_img]
                 except Exception as e:
                      print(f"    Error pasting image {i}: {e}")
                      traceback.print_exc()
        print("Stitching complete.")
        return result


def notify(title, message):
    subprocess.run(['notify-send', title, message])

def get_longscreenshot_path():
    path = Path.home() / "Pictures/Longscreenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path

def cleanup():
    if LOCK_FILE.exists():
        print("LOG: Cleaning up lock file.")
        LOCK_FILE.unlink()


def extractor(video_path, frame_number, temp_dir, fps):
    output_filename = temp_dir / f"frame_{frame_number}.png"

    while not video_path.exists():
        time.sleep(0.2)

    timestamp = (frame_number - 1) / fps
    extract_cmd = ['ffmpeg', '-y', '-ss', str(timestamp), '-i', str(video_path), '-vframes', '1', str(output_filename)]
    result = subprocess.run(extract_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode == 0 and output_filename.exists() and output_filename.stat().st_size > 0:
        return cv2.imread(str(output_filename))
    else:
        return None

def wait_for_file_write_finish(video_path, stability_checks=3, check_interval=0.25):
    """
    Waits for a file to stop growing in size, indicating it's fully written.
    """
    print(f"LOG: Waiting for video file '{video_path.name}' to be finalized...")
    last_size = -1
    stable_count = 0
    
    while stable_count < stability_checks:
        try:
            current_size = video_path.stat().st_size
            if current_size == last_size and current_size > 0:
                stable_count += 1
            else:
                stable_count = 0  # Reset if size changes
            
            last_size = current_size
        except FileNotFoundError:
            # File might not exist yet, keep waiting
            stable_count = 0
            last_size = -1
        
        # Wait before the next check
        time.sleep(check_interval)
        
    print("LOG: File size has stabilized. Proceeding with frame counting.")

def send_stop_signal_to_controller():
    """Send SIGTERM to the controller process to trigger finalization"""
    try:
        if LOCK_FILE.exists():
            controller_pid = int(LOCK_FILE.read_text())
            print(f"LOG: Stitcher: Sending stop signal to controller PID {controller_pid}")
            os.kill(controller_pid, signal.SIGTERM)
            return True
    except (ValueError, ProcessLookupError, FileNotFoundError) as e:
        print(f"LOG: Stitcher: Failed to send stop signal to controller: {e}")
        return False
    return False

def get_frame_count_with_retry(video_path, max_retries=10, delay=0.1):
    """
    Try to get frame count from video file with retries and delay.
    This handles cases where the video file is still being written to.
    """
    # --- METHOD 1: Fast Packet Counting (Primary) ---
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_packets', '-show_entries', 'stream=nb_read_packets',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        total_frames = int(probe.stdout.strip())
        if total_frames > 0:
            print(f"LOG: Successfully got packet count (fast method): {total_frames}")
            return total_frames
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"LOG: Fast packet counting failed, trying next method. Error: {e}")

    # --- METHOD 2: Slower Frame Counting (Fallback) ---
    for attempt in range(max_retries):
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                str(video_path)
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            total_frames = int(probe.stdout.strip())
            print(f"LOG: Successfully got frame count (fallback method): {total_frames} (attempt {attempt + 1})")
            return total_frames

        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"LOG: ffprobe frame count attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"LOG: Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                print("LOG: All ffprobe attempts failed. Video may be corrupted or still being written.")
                # --- METHOD 3: Alternative ffmpeg parsing (Last Resort) ---
                try:
                    print("LOG: Trying alternative frame counting method...")
                    alt_cmd = [
                        'ffmpeg', '-i', str(video_path), '-map', '0:v:0',
                        '-c', 'copy', '-f', 'null', '-'
                    ]
                    result = subprocess.run(alt_cmd, capture_output=True, text=True, check=False)
                    # Parse ffmpeg output for frame count
                    for line in result.stderr.split('\n'):
                        if 'frame=' in line:
                            frame_part = line.split('frame=')[1].split()[0]
                            try:
                                total_frames = int(frame_part)
                                print(f"LOG: Alternative method found {total_frames} frames")
                                return total_frames
                            except ValueError:
                                continue
                except Exception as alt_e:
                    print(f"LOG: Alternative frame counting also failed: {alt_e}")

    print("LOG: Could not determine frame count after all attempts")
    return -1

def stitcher_process(video_path, ipc_queue):
    global JUMP
    print("LOG: Stitcher process started.")
    temp_dir = Path(f"/tmp/longscreenshot_frames_{os.getpid()}")
    temp_dir.mkdir(exist_ok=True)

    final_frame_count = None
    frame_cache = {}
    fps = ipc_queue.get() # Get fps from queue

    while 1 not in frame_cache:
        print("LOG: Stitcher: Waiting for first frame...")
        frame = extractor(video_path, 1, temp_dir, fps)
        if frame is not None:
            frame_cache[1] = frame
        else:
            time.sleep(0.5)

    stitcher = Stitcher()
    stitcher.add_image(frame_cache[1])

    last_successful_idx = 1

    while True:
        if final_frame_count and last_successful_idx >= final_frame_count:
            print("LOG: Stitcher: Reached end of video.")
            break

        if final_frame_count is None:
            try:
                # Non-blocking check for the final frame count
                final_frame_count = ipc_queue.get_nowait()
                if final_frame_count > 0: 
                    print(f"LOG: Stitcher: Received final frame count: {final_frame_count}")
            except Exception: pass
        else:
            subprocess.run(
                ["zenity", "--notification", "--text", f"Have stitched frame {last_successful_idx}/{final_frame_count}."],
                stderr=subprocess.DEVNULL
            )
        
        lower_bound = last_successful_idx
        
        # --- Start of Revised Section ---
        
        # This loop attempts to find the next matching frame using a binary search approach.
        # It starts by trying to match a frame JUMP positions away. If that fails,
        # it tries the frame in the middle of the current position and the failed attempt,
        # repeating until a match is found or the search space is exhausted.
        
        search_upper = lower_bound + JUMP
        if final_frame_count:
            search_upper = min(search_upper, final_frame_count)

        if search_upper <= lower_bound:
            break

        found_next_match = False
        
        # The search loop continues as long as there's a gap to check.
        while search_upper > lower_bound:
            # Check for the final frame count from the controller process again inside the loop
            if final_frame_count is None:
                try:
                    final_frame_count = ipc_queue.get_nowait()
                    if final_frame_count > 0:
                        print(f"LOG: Stitcher: Received final frame count during search: {final_frame_count}")
                        # Adjust the search boundary if we now have the final count
                        search_upper = min(search_upper, final_frame_count)
                except Exception: pass

            target_idx = search_upper
            print(f"LOG: Stitcher: Trying to match with frame {target_idx}...")

            # Extract the frame if it's not already in our cache
            if target_idx not in frame_cache:
                frame = extractor(video_path, target_idx, temp_dir, fps)
                if frame is not None:
                    frame_cache[target_idx] = frame
                else:
                    # If frame extraction fails, it could be because the recording hasn't finished writing this frame yet.
                    if final_frame_count:
                        # If we have the final frame count, this is an unexpected error.
                        search_upper = lower_bound # Collapse the search space to stop.
                        break
                    else:
                        # Otherwise, wait and try again.
                        time.sleep(0.2)
                        continue
            
            # Check if adding the image increases the count of stitched images.
            # This is the key to determining if a match was successful.
            images_before = len(stitcher.images)
            stitcher.add_image(frame_cache[target_idx])
            images_after = len(stitcher.images)

            if images_after > images_before:
                # SUCCESS: A new, unique frame was matched and added.
                print(f"LOG: Stitcher: SUCCESS. Matched and added frame {target_idx}.")
                last_successful_idx = target_idx
                found_next_match = True
                break # Exit the search loop and start a new JUMP from the new position.
            else:
                # FAILURE: The frame did not match.
                print(f"LOG: Stitcher: FAILED to match frame {target_idx}. Reducing search distance.")
                
                # If we failed on the frame immediately following the last success, we can't proceed.
                if search_upper - lower_bound <= 1:
                    print("LOG: Stitcher: Cannot match consecutive frames. Sending stop signal to controller.")
                    print(f"LOG: Stitcher: Last successful frame: {last_successful_idx}")
                    print(f"LOG: Stitcher: Failed frame: {search_upper}")
                    print("LOG: Stitcher: This typically indicates:")
                    print("  - Content changed significantly (new page/window)")
                    print("  - Recording stopped or paused")
                    print("  - Video encoding issues")
                    print("  - Excessive motion blur or compression artifacts")
                    
                    # ADDED: Send stop signal to controller when consecutive frames can't be matched
                    if final_frame_count is None:  # Only send signal if recording is still active
                        send_stop_signal_to_controller()
                        # Wait for the controller to send the final frame count
                    
                    search_upper = lower_bound # Collapse search space to exit loop.
                    break
 
                # This is the bisection: try the frame in the middle of the last success and the current failure.
                search_upper = (lower_bound + search_upper) // 2
                JUMP=search_upper-lower_bound
        
        # --- End of Revised Section ---

        if not found_next_match:
            print("LOG: Stitcher: No further matches found in the current search.")
            break

    # Replace this section in your stitcher_process function:
    final_image = stitcher.stitch()
    if final_image is not None:
        save_path = get_longscreenshot_path() / f"longscreenshot_{int(time.time())}.png"
        cv2.imwrite(str(save_path), final_image)
        
        try:
            # Copy to clipboard using wl-copy for Wayland
            with open(save_path, 'rb') as f:
                process = subprocess.Popen(['wl-copy', '--type', 'image/png'], stdin=subprocess.PIPE)
                process.communicate(input=f.read())
            
            if process.returncode == 0:
                notify("Success", f"Long screenshot saved to {save_path} and copied to clipboard.")
            else:
                # Fallback: try xclip for X11 systems
                try:
                    img = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
                    output = io.BytesIO()
                    img.save(output, format='PNG')
                    process = subprocess.Popen(['xclip', '-selection', 'clipboard', '-t', 'image/png'], stdin=subprocess.PIPE)
                    process.communicate(input=output.getvalue())
                    notify("Success", f"Long screenshot saved to {save_path} and copied to clipboard.")
                except Exception as e:
                    notify("Success", f"Saved to {save_path}. Clipboard copy failed: {e}")
                    print(f"LOG: Clipboard copy failed: {e}")
        except Exception as e:
            notify("Success", f"Saved to {save_path}. Clipboard copy failed: {e}")
            print(f"LOG: Clipboard copy failed: {e}")

    for f in temp_dir.glob('*.png'): f.unlink()
    temp_dir.rmdir()
    print("LOG: Stitcher process finished.")

def controller_signal_handler(signum, frame, stitcher_proc, ipc_queue):
    print("\nLOG: Controller: Stop signal received.")
    cl = obs.ReqClient()
    if cl.get_record_status().output_active:
        cl.stop_record()
    cl.disconnect()
    
    try:
        screencast_dir = Path.home() / "Videos/Screencasts"
        video_path = max(screencast_dir.glob('*.mkv'), key=os.path.getmtime)
        
        # --- NEW CODE ---
        # Wait for OBS to finish writing the file before trying to read it.
        wait_for_file_write_finish(video_path)
        # --- END NEW CODE ---

        # Now, this function is much more likely to succeed on the first try.
        total_frames = get_frame_count_with_retry(video_path)
        
        if total_frames > 0:
            ipc_queue.put(total_frames)
            print(f"LOG: Controller: Sent final frame count ({total_frames}) to stitcher.")
        else:
            print("LOG: Controller: Could not determine frame count, sending -1 to stitcher.")
            ipc_queue.put(-1)
        
    except Exception as e:
        print(f"LOG: Controller: Error getting frame count: {e}")
        ipc_queue.put(-1)

    stitcher_proc.join(timeout=120)
    cleanup()
    sys.exit(0)

def main():
    if LOCK_FILE.exists():
        try:
            subprocess.run(["pkill","slop"], check=False)
            pid = int(LOCK_FILE.read_text())
            print(f"LOG: Lock file found. Sending stop signal to PID {pid}.")
            os.kill(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError) as e:
            print(f"LOG: Stale lock file found. Removing. Error: {e}")
        cleanup()
        sys.exit(0)

    atexit.register(cleanup)
    LOCK_FILE.write_text(str(os.getpid()))

    if subprocess.run(["pgrep", "-x", "obs"], capture_output=True).returncode == 0:
        print("LOG: OBS is running.")
    else:
        print("LOG: OBS not running. Starting it now.")
        subprocess.Popen(["obs"])
        sys.exit(1)

    try:
        proc = subprocess.run(["slop", "-f", "%w %h %x %y"], capture_output=True, text=True, check=True)
        w, h, x, y = [int(v) for v in proc.stdout.strip().split()]
        if w < 32 or h < 32:
            notify("Error", "Selected region is too small.")
            sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(0)

    try:
        cl = obs.ReqClient(host="localhost", port=4455, password="", timeout=10)
        
        video_settings = cl.get_video_settings()
        video_fps_num = video_settings.fps_numerator
        video_fps_den = video_settings.fps_denominator
        fps = video_fps_num / video_fps_den

        filter_settings = {
            "top": y, 
            "left": x, 
            "bottom": max(0, video_settings.base_height - (y + h)), 
            "right": max(0, video_settings.base_width - (x + w))
        }
        filter_name = "ScriptedCrop"
        cl.set_source_filter_settings(OBS_SOURCE_NAME, filter_name, filter_settings)
        cl.set_source_filter_enabled(OBS_SOURCE_NAME, filter_name, True)

        cl.set_video_settings(
            base_width=w, 
            base_height=h, 
            out_width=w, 
            out_height=h, 
            numerator=video_fps_num, 
            denominator=video_fps_den
        )

        scene_response = cl.get_current_program_scene()
        scene_name = scene_response.current_program_scene_name
        
        id_response = cl.get_scene_item_id(scene_name, OBS_SOURCE_NAME)
        item_id = id_response.scene_item_id

        reset_transform = {
            "positionX": 0.0,
            "positionY": 0.0,
            "rotation": 0.0,
            "scaleX": 1.0,
            "scaleY": 1.0,
            "cropTop": 0,
            "cropBottom": 0,
            "cropLeft": 0,
            "cropRight": 0,
        }
        cl.set_scene_item_transform(scene_name, item_id, reset_transform)
        print("LOG: Source transform has been reset to fill the new canvas.")

        if cl.get_record_status().output_active:
            notify("Error", "OBS is already recording.")
            sys.exit(1)

        cl.start_record()
        time.sleep(1)

        screencast_dir = Path.home() / "Videos/Screencasts"
        video_path = max(screencast_dir.glob('*.mkv'), key=os.path.getmtime)

        ipc_queue = Queue()
        ipc_queue.put(fps) # Put fps in queue
        stitcher_p = Process(target=stitcher_process, args=(video_path, ipc_queue))
        stitcher_p.start()

        signal.signal(signal.SIGTERM, lambda s, f: controller_signal_handler(s, f, stitcher_p, ipc_queue))
        print("LOG: Controller is now paused, waiting for stop signal.")
        signal.pause()

    except Exception as e:
        notify("Video Recording Error", f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

