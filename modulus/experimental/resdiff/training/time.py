import datetime
import cftime

def convert_datetime_to_cftime(time: datetime.datetime, cls=cftime.DatetimeGregorian) -> cftime.DatetimeGregorian:
    return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)
