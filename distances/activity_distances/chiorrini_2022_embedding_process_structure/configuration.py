from _csv import writer

def import_path(x):
    if x == 1:
        net_path = "log/BPI2012_SE_n10og.pnml"
    elif x == 2:
        net_path =  "log/BPI2017_FilterOnComplete.pnml"

    return net_path


def file_print(path, text, mode='a'):
    if type(text) is list or type(text) is tuple:
        string = ''
        for t in text:
            string += str(t) + ';'
    else:
        string = str(text)
    string = string.replace('.', ',')
    with open(path, mode, newline='') as filep:
        writer_object = writer(filep, delimiter = ";")
        writer_object.writerow([string])
        filep.close()