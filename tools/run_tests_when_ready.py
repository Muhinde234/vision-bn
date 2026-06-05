import time
import sys
import urllib.request
import subprocess

HOST = 'http://localhost:8000/health'
WAIT_SECS = 60

for i in range(WAIT_SECS):
    try:
        r = urllib.request.urlopen(HOST, timeout=2)
        if r.getcode() == 200:
            print('Server ready')
            break
    except Exception as e:
        print('.', end='', flush=True)
        time.sleep(1)
else:
    print('\nServer did not become ready within timeout')
    sys.exit(2)

# Run tests
rc = subprocess.call([sys.executable, 'test_all_apis.py'])
print('Test runner exit code', rc)
sys.exit(rc)
