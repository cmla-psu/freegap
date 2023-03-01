import argparse
from pathlib import Path
from freegap import compress_pdfs

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('pdf', help=f'The pdf files.', nargs='+')
    results = arg_parser.parse_args()
    for path in results.pdf:
        pdf_file = Path(path)
        compress_pdfs([str(pdf_file.resolve())])
