import csv
import os


class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.header = None
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    def initLogHeaders(self, header):
        """
        Initialize or overwrite the CSV file with the given header.
        """
        self.header = header
        with open(self.filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(self.header)

    def write_row(self, row):
        """
        Write a row to the CSV file.
        """
        if not self.header:
            raise RuntimeError(
                "Header not initialized. Call initLogHeaders() first.")
        if len(row) != len(self.header):
            raise ValueError(f"Row must have {len(self.header)} elements.")
        with open(self.filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def writeData(self, data_dict):
        """
        Write data to the CSV file based on dictionary keys.
        If the value is a list, it will spread across columns.
        """
        if not self.header:
            raise RuntimeError(
                "Header not initialized. Call initLogHeaders() first.")

        row = [''] * len(self.header)
        used_indices = set()

        for key, value in data_dict.items():
            if key in self.header:
                index = self.header.index(key)
                if isinstance(value, list):
                    # Spread list values starting from this index
                    for i, v in enumerate(value):
                        if index + i < len(self.header):
                            row[index + i] = v
                            used_indices.add(index + i)
                else:
                    row[index] = value
                    used_indices.add(index)

        self.write_row(row)
