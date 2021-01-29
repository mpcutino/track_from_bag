from PyQt4 import QtGui


def show_message(message, title="Error!"):
    e = QtGui.QMessageBox()
    e.setWindowTitle(title)
    e.setText(message)
    e.exec_()
