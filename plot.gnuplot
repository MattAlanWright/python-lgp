set datafile separator ','
set key autotitle columnhead
set ylabel 'Score percentage'
set xlabel 'Number of generations'
plot 'file.csv' using 1:2 with lines, '' using 1:3 with lines
