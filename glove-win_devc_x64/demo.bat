@ECHO OFF
:: This batch file reveals OS, hardware, and networking configuration.
TITLE My System Info
ECHO Please wait... Checking system information.
:: Section 1: OS information.
ECHO ============================
ECHO OS INFO
ECHO ============================
systeminfo | findstr /c:"OS Name"
systeminfo | findstr /c:"OS Version"
systeminfo | findstr /c:"System Type"
:: Section 2: Hardware information.
ECHO ============================
ECHO HARDWARE INFO
ECHO ============================

SET CORPUS=donald.txt
SET VOCAB_FILE=vocab.txt
SET COOCCURRENCE_FILE=cooccurrence.bin
SET COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
SET SAVE_FILE=vectors
SET VERBOSE=2
SET MEMORY=4.0
SET VOCAB_MIN_COUNT=5
SET VECTOR_SIZE=50
SET MAX_ITER=15
SET WINDOW_SIZE=15
SET BINARY=2
SET NUM_THREADS=8
SET X_MAX=10

SET PYTHON=C:\Users\minsong\AppData\Local\Programs\Python\Python37\python.exe

ECHO vocab_count -min-count %VOCAB_MIN_COUNT% -verbose %VERBOSE% < %CORPUS% > %VOCAB_FILE%
vocab_count -min-count %VOCAB_MIN_COUNT% -verbose %VERBOSE% < %CORPUS% > %VOCAB_FILE%

ECHO cooccur -memory %MEMORY% -vocab-file %VOCAB_FILE% -verbose %VERBOSE% -window-size %WINDOW_SIZE% < %CORPUS% > %COOCCURRENCE_FILE%
cooccur -memory %MEMORY% -vocab-file %VOCAB_FILE% -verbose %VERBOSE% -window-size %WINDOW_SIZE% < %CORPUS% > %COOCCURRENCE_FILE%

ECHO shuffle -memory %MEMORY% -verbose %VERBOSE% < %COOCCURRENCE_FILE% > %COOCCURRENCE_SHUF_FILE%
shuffle -memory %MEMORY% -verbose %VERBOSE% < %COOCCURRENCE_FILE% > %COOCCURRENCE_SHUF_FILE%

ECHO glove -save-file %SAVE_FILE% -threads %NUM_THREADS% -input-file %COOCCURRENCE_SHUF_FILE% -x-max %X_MAX% -iter %MAX_ITER% -vector-size %VECTOR_SIZE% -binary %BINARY%0 -vocab-file %VOCAB_FILE% -verbose %VERBOSE%
glove -save-file %SAVE_FILE% -threads %NUM_THREADS% -input-file %COOCCURRENCE_SHUF_FILE% -x-max %X_MAX% -iter %MAX_ITER% -vector-size %VECTOR_SIZE% -binary %BINARY%0 -vocab-file %VOCAB_FILE% -verbose %VERBOSE%

ECHO %PYTHON% eval/python/evaluate.py
%PYTHON% eval/python/evaluate.py
