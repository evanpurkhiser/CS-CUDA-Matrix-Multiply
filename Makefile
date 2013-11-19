report.pdf: report.md
	pandoc report.md -V geometry:margin=1in -o report.pdf

clean:
	rm -f {8x8,16x16}/{dnsmb1,*.txt}
	rm -f report.pdf
