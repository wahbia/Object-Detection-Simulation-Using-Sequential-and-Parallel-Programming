#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#define NUM_THREADS 8
#define NUM_OBJECTS 90000000

// Global variables to keep track of detected objects and confidence scores for different scenarios.
int detected_objects_without_race_handling = 0;
int detected_objects_with_race_handling  = 0;
int detected_objects_sequential  = 0;

double confidence_score_without_race_handling  = 0.0;
double confidence_score_with_race_handling  = 0.0;
double confidence_score_sequential = 0.0;

// Arrays to store the results of object detection and confidence scores.
int detection_results[NUM_OBJECTS];
float confidence_scores[NUM_OBJECTS];

// Function to initialize the arrays with random values.
void initialize_arrays()
{
    for (int i = 0; i < NUM_OBJECTS; ++i)
    {
        // Randomly determine if an object is detected (0 or 1) and assign a random confidence score.
        detection_results[i] = rand() % 2;
        confidence_scores[i] = (float)rand() / (float)(RAND_MAX);
    }
}

// Simulated function to detect objects in the image.
int detect_object(int obj_id) {
    // Returns the detection result (0 or 1) for the given object ID.
    return detection_results[obj_id];
}

// Simulated function to compute confidence score.
double compute_confidence_score(int obj_id) {
    // Returns the confidence score for the given object ID.
    return confidence_scores[obj_id];
}

// Function to print the detection results.
void print_detection_results() {
    // Print the detection results and confidence scores for each approach.
    printf("\nDetection results:\n");
    printf("Total detected objects (with race condition handling): %d\n", detected_objects_without_race_handling);
    printf("Total confidence score (with race condition handling): %.2f\n", confidence_score_without_race_handling );
    printf("Total detected objects (without race condition handling): %d\n", detected_objects_with_race_handling );
    printf("Total confidence score (without race condition handling): %.2f\n", confidence_score_with_race_handling );
    printf("Total detected objects (sequential): %d\n", detected_objects_sequential);
    printf("Total confidence score (sequential): %.2f\n", confidence_score_sequential);
}

// Function to write the detection results to a file.
void write_detection_results_to_file(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the detection results and confidence scores to the file.
    fprintf(file, "Detection results:\n");
    fprintf(file, "Total detected objects (without race condition handling): %d\n", detected_objects_without_race_handling );
    fprintf(file, "Total confidence score (without race condition handling): %.2f\n", confidence_score_without_race_handling );
    fprintf(file, "Total detected objects (with race condition handling): %d\n", detected_objects_with_race_handling );
    fprintf(file, "Total confidence score (with race condition handling): %.2f\n", confidence_score_with_race_handling );
    fprintf(file, "Total detected objects (sequential): %d\n", detected_objects_sequential);
    fprintf(file, "Total confidence score (sequential): %.2f\n", confidence_score_sequential);

    fclose(file);
}

int main() {
    // Seed the random number generator.
    srand(42);
    // Initialize arrays with random detection results and confidence scores.
    initialize_arrays();

    // Variables to store start and end times for performance measurement.
    double start_time, end_time;

    // Parallelized object detection with race condition handling.
    printf("\nRunning object detection with race condition handling:\n");
    start_time = omp_get_wtime();
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for 
    for (int i = 0; i < NUM_OBJECTS; i++)
    {
        // Detect object and update global counters.
        // This approach does not use reduction or critical sections, which can lead to a race condition.
        int is_detected = detect_object(i);
        if (is_detected)
        {
            detected_objects_with_race_handling ++;
            confidence_score_with_race_handling  += compute_confidence_score(i);
        }
    }
    end_time = omp_get_wtime();
    double execution_time_with_race_handling  = end_time - start_time;

    // Parallelized object detection without race condition handling using reduction.
    printf("\nRunning object detection without race condition handling:\n");
    start_time = omp_get_wtime();
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for //reduction(+:detected_objects_without_race_handling , confidence_score_without_race_handling )
    for (int i = 0; i < NUM_OBJECTS; i++)
    {
        // Detect object and update global counters using OpenMP reduction to avoid race conditions.
        int is_detected = detect_object(i);
        if (is_detected)
         #pragma omp critical

        {
            detected_objects_without_race_handling ++;
            confidence_score_without_race_handling  += compute_confidence_score(i);
        }
    }
    end_time = omp_get_wtime();
    double execution_time_without_race_handling  = end_time - start_time;

    // Sequential object detection.
    printf("\nRunning sequential object detection:\n");
    start_time = omp_get_wtime();
    for (int i = 0; i < NUM_OBJECTS; i++)
    {
        // Detect object and update global counters sequentially.
        int is_detected = detect_object(i);
        if (is_detected)
        {
            detected_objects_sequential++;
            confidence_score_sequential += compute_confidence_score(i);
        }
    }
    end_time = omp_get_wtime();
    double execution_time_sequential = end_time - start_time;

    // Print and write detection results to file.
    print_detection_results();
    write_detection_results_to_file("detection_results.txt");

    // Calculate and display execution times and efficiency.
    printf("\nRuntime with race condition handling: %.2f seconds\n", execution_time_with_race_handling );
    printf("Runtime without race condition handling: %.2f seconds\n", execution_time_without_race_handling );
    printf("Runtime sequential: %.2f seconds\n", execution_time_sequential);

    double speedup, efficiency;
    speedup = execution_time_sequential / execution_time_without_race_handling ;
    printf("Speedup without race handling: %.2f\n", speedup);
    efficiency = speedup / NUM_THREADS;
    printf("Efficiency without race handling: %.2f\n", efficiency);

    speedup = execution_time_sequential / execution_time_with_race_handling ;
    printf("Speedup with race handling: %.2f\n", speedup);
    efficiency = speedup / NUM_THREADS;
    printf("Efficiency with race handling: %.2f\n", efficiency);

    return 0;
}
