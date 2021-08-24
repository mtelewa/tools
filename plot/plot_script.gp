set term postscript enhanced size 20cm,15cm landscape color
set output 'timestep-energy.eps' 
set encoding iso_8859_1

#unset key
#set xrange [0:500]
set border lw 1
#set title " g(r) for Al in different crystal structures" 
set xlabel " time step"
set ylabel " Energy (Kcal/mol)"
#set grid
set key Right bottom inside
#set title font "Calibri,20"
set xlabel font "Helvetica,11"
set ylabel font "Helvetica,11" offset 1
set xtics font "Helvetica, 11"
set ytics font "Helvetica, 11"
set key font "Helvetica, 11"
#set border 10 back ls 80
set lmargin 10
set rmargin 10
set tmargin 5
set bmargin 5


plot "01/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "aquamarine" title "{/Symbol D}t=0.1 fs", \
"05/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "black" title "{/Symbol D}t=0.5 fs", \
"1/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "blue" title "{/Symbol D}t=1.0 fs", \
"2/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "violet" title "{/Symbol D}t=2.0 fs", \
"3/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "red" title "{/Symbol D}t=3.0 fs", \
"4/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "green" title "{/Symbol D}t=4.0 fs", \
"5/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "grey" title "{/Symbol D}t=5.0 fs"
#"10/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "violet" title "{/Symbol D}t=10.0 fs"
#"20/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "orange" title "{/Symbol D}t=20.0 fs"
#"50/thermo.out" u ($1/1e6):5 w l lt 1 lw 2 lc rgb "black" title "{/Symbol D}t=50.0 fs"
