"""Coordinator-only ledger. Dispatch, settlement, budget and stops are durable."""
import json

from .common import canonical, connect, require


class BuildState:
    def __init__(self, root, catalog):
        self.db = connect(root / 'state.sqlite')
        self.catalog = catalog
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS identity(build_id TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS tasks(task_id INTEGER PRIMARY KEY, members TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING', retry_epoch INTEGER NOT NULL DEFAULT 0,
            retry_count INTEGER NOT NULL DEFAULT 0, output TEXT);
          CREATE TABLE IF NOT EXISTS pools(pool_id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL, failure_id TEXT UNIQUE, startup_failed INTEGER NOT NULL DEFAULT 0,
            reason TEXT);
          CREATE TABLE IF NOT EXISTS dispatches(attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES tasks(task_id), pool_id INTEGER NOT NULL REFERENCES pools(pool_id),
            retry_epoch INTEGER NOT NULL, settled INTEGER NOT NULL DEFAULT 0, started INTEGER,
            ending TEXT, error TEXT, failure_delta INTEGER, revoked INTEGER NOT NULL DEFAULT 0,
            UNIQUE(task_id,attempt_id));
          CREATE UNIQUE INDEX IF NOT EXISTS one_unsettled ON dispatches(task_id) WHERE settled=0;
          CREATE TABLE IF NOT EXISTS stops(code TEXT PRIMARY KEY, detail TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS resolutions(resolution_id INTEGER PRIMARY KEY,
            action TEXT NOT NULL, task_id INTEGER, reason TEXT NOT NULL);
        ''')
        with self.db:
            identity = self.db.execute('SELECT build_id FROM identity').fetchone()
            if identity is None:
                self.db.execute('INSERT INTO identity VALUES (?)', (catalog['build_id'],))
                for task in catalog['tasks']:
                    self.db.execute('INSERT INTO tasks(task_id,members) VALUES (?,?)',
                                    (task['task_id'], canonical(task['members']).decode()))
            else:
                require(identity[0] == catalog['build_id'], 'BINDING_MISMATCH', 'state build')
                require([(r['task_id'], json.loads(r['members'])) for r in self.db.execute('SELECT * FROM tasks ORDER BY task_id')]
                        == [(t['task_id'], t['members']) for t in catalog['tasks']], 'BINDING_MISMATCH', 'state members')

    def close(self):
        self.db.close()

    def stops(self):
        return {r['code']: json.loads(r['detail']) for r in self.db.execute('SELECT * FROM stops')}

    def stop(self, code, detail):
        self.db.execute('INSERT OR IGNORE INTO stops VALUES (?,?)', (code, canonical(detail).decode()))

    def new_pool(self):
        require(not self.stops(), 'BUILD_STOPPED', 'explicit resume required')
        require(not self.db.execute("SELECT 1 FROM pools WHERE status!='CLOSED'").fetchone(),
                'UNRECONCILED_POOL', 'old pool must be closed')
        with self.db:
            cursor = self.db.execute("INSERT INTO pools(status) VALUES ('OPEN')")
        return cursor.lastrowid

    def dispatch(self, task_id, pool_id):
        require(not self.stops(), 'BUILD_STOPPED', 'no dispatch during stop')
        with self.db:
            task = self.db.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
            require(task['status'] == 'PENDING' and task['retry_count'] < 3,
                    'RETRY_EXHAUSTED', str(task_id))
            require(self.db.execute('SELECT status FROM pools WHERE pool_id=?', (pool_id,)).fetchone()[0] == 'OPEN',
                    'STALE_POOL', str(pool_id))
            cursor = self.db.execute('INSERT INTO dispatches(task_id,pool_id,retry_epoch) VALUES (?,?,?)',
                                    (task_id, pool_id, task['retry_epoch']))
            self.db.execute("UPDATE tasks SET status='RUNNING' WHERE task_id=?", (task_id,))
        return dict(task_id=task_id, pool_id=pool_id, retry_epoch=task['retry_epoch'], attempt_id=cursor.lastrowid)

    def reconciling(self, pool_id, *, startup_failed=False, reason=None):
        with self.db:
            self.db.execute("UPDATE pools SET status='RECONCILING',failure_id=COALESCE(failure_id,?),startup_failed=MAX(startup_failed,?),reason=COALESCE(reason,?) WHERE pool_id=? AND status!='CLOSED'",
                (f'pool-{pool_id}', int(startup_failed), reason, pool_id))

    def settle(self, attempt_id, *, started, ending, error=None, complete=False, hook=None):
        """Caller holds task lock. Never hold this transaction waiting for a lock."""
        with self.db:
            attempt = self.db.execute('SELECT * FROM dispatches WHERE attempt_id=?', (attempt_id,)).fetchone()
            require(attempt is not None, 'ATTEMPT_CONFLICT', str(attempt_id))
            if attempt['settled']:
                return dict(attempt)
            delta = int(ending in ('INTERRUPTED', 'RETRYABLE_FAILURE'))
            self.db.execute('UPDATE dispatches SET settled=1,started=?,ending=?,error=?,failure_delta=?,revoked=? WHERE attempt_id=?',
                (int(started), ending, canonical(error).decode(), delta, int(not started and not complete), attempt_id))
            task = self.db.execute('SELECT * FROM tasks WHERE task_id=?', (attempt['task_id'],)).fetchone()
            require(task['retry_epoch'] == attempt['retry_epoch'], 'ATTEMPT_CONFLICT', 'epoch changed before settlement')
            count = task['retry_count'] + delta
            status = 'READY' if complete else ('FAILED' if count >= 3 or ending == 'REQUIRES_RESUME' else 'PENDING')
            self.db.execute('UPDATE tasks SET status=?,retry_count=? WHERE task_id=?',
                            (status, count, attempt['task_id']))
            if ending == 'REQUIRES_RESUME':
                self.stop((error or {}).get('code', 'REQUIRES_RESUME'), error or {})
            if ending == 'STOPPED':
                self.stop('USER_STOP', error or {})
            if hook:
                hook('SETTLEMENT_WRITTEN',dict(attempt))
            return dict(self.db.execute('SELECT * FROM dispatches WHERE attempt_id=?', (attempt_id,)).fetchone())

    def close_pool(self, pool_id):
        with self.db:
            require(not self.db.execute('SELECT 1 FROM dispatches WHERE pool_id=? AND settled=0', (pool_id,)).fetchone(),
                    'UNRECONCILED_POOL', str(pool_id))
            pool = self.db.execute('SELECT * FROM pools WHERE pool_id=?', (pool_id,)).fetchone()
            pending = self.db.execute("SELECT 1 FROM tasks WHERE status!='DONE'").fetchone() is not None
            started = self.db.execute('SELECT 1 FROM dispatches WHERE pool_id=? AND started=1', (pool_id,)).fetchone() is not None
            if pending and pool['status'] == 'RECONCILING' and (pool['startup_failed'] or not started):
                self.stop('POOL_START_FAILED' if pool['startup_failed'] else 'POOL_NO_START',
                          dict(pool_id=pool_id, failure_id=pool['failure_id'], reason=pool['reason']))
            self.db.execute("UPDATE pools SET status='CLOSED' WHERE pool_id=?", (pool_id,))

    def explicit_resume(self, reason, *, retry_tasks=()):
        require(type(reason) is str and reason.strip(), 'MISSING_FIELD', 'explicit resume reason')
        require(not self.db.execute('SELECT 1 FROM dispatches WHERE settled=0').fetchone(),
                'UNRECONCILED_ATTEMPT', 'reconcile before explicit resume')
        with self.db:
            for row in self.db.execute('SELECT * FROM stops').fetchall():
                self.db.execute('INSERT INTO resolutions(action,reason) VALUES (?,?)',
                                ('resume:' + row['code'], reason))
            self.db.execute('DELETE FROM stops')
            # Requires-resume failures keep their current epoch/count.
            self.db.execute("UPDATE tasks SET status='PENDING' WHERE status='FAILED' AND retry_count<3")
            for task_id in retry_tasks:
                row = self.db.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
                require(row is not None and row['status'] != 'DONE', 'INVALID_REFERENCE', 'retry task')
                self.db.execute('INSERT INTO resolutions(action,task_id,reason) VALUES (?,?,?)',
                                ('new-retry-epoch', task_id, reason))
                self.db.execute("UPDATE tasks SET retry_epoch=retry_epoch+1,retry_count=0,status='PENDING' WHERE task_id=?", (task_id,))
