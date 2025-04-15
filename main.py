import csv


class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.headers = []

    def initLogHeaders(self, headers):
        # Initialize CSV file with headers
        self.headers = headers
        with open(self.filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    def writeData(self, **kwargs):
        # Dynamically write a row based on the provided keyword arguments and headers
        data_row = [kwargs.get(header, "") for header in self.headers]
        with open(self.filename, mode='a', newline='') as file:
            csv.writer(file).writerow(data_row)

    def writeRangeData(self, **kwargs):
        # Dynamically handle ranges by determining the max length of provided ranges
        max_len = max(len(value) if isinstance(value, (list, tuple))
                      else 1 for value in kwargs.values())

        # For each range, loop through its values and create a row for each iteration
        for i in range(max_len):
            data_row = {key: (value[i % len(value)] if isinstance(value, (list, tuple)) else value)
                        for key, value in kwargs.items()}
            self.writeData(**data_row)


def main():
    # Initialize the CSV logger
    TestCSV = CSVLogger("C:/Nagashree/Python files/test.csv")

    # Initialize headers for the CSV file (with new header names)
    TestCSV.initLogHeaders(["Cycle_Count", "temperature", "config", "data1",
                           "data2", "data3", "data4", "data5", "data6", "data7", "data8"])

    # Write data with dynamic column handling
    TestCSV.writeRangeData(Cycle_Count=1, temperature=2,
                           config="xml1", data1=(1, 3, 4, 6, 7))
    TestCSV.writeRangeData(Cycle_Count=2, temperature=30, config="xml3", data4=(
        1, 3, 4, 6, 7), data5=(1, 3, 4, 6, 7), data6=(1, 3, 4, 6))
    TestCSV.writeRangeData(Cycle_Count=3, temperature6=40, config="xml6", data6=(
        1, 3, 4, 6, 7), data5=(1, 3, 4, 6, 7), data7=(1, 3, 4, 6, 7, 8, 9, 0))


if __name__ == "__main__":
    main()
