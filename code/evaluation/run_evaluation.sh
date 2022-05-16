echo run evaluation compute_MPI_error_revisiting_video
matlab -nodisplay -nosplash -nodesktop -r "run('./compute_MPI_error_revisiting_video.m');exit;"

echo run evaluation compute_MPI_error_video
matlab -nodisplay -nosplash -nodesktop -r "run('./compute_MPI_error_video.m');exit;"

echo run evaluation compute_MIT_error.m
matlab -nodisplay -nosplash -nodesktop -r "run('./compute_MIT_error.m');exit;"