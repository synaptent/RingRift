/*
 * RingRift Sentinel - Minimal Process Supervision
 *
 * This C binary monitors the watchdog heartbeat file and restarts the
 * watchdog process if the heartbeat goes stale. It's designed to be:
 *
 * 1. Minimal - No Python dependencies, survives interpreter crashes
 * 2. Robust - Runs via launchd/systemd, auto-restarted by OS
 * 3. Simple - Just checks file mtime, no complex logic
 *
 * Architecture:
 *   launchd/systemd (OS-level, always running)
 *       |
 *       v KeepAlive
 *   ringrift_sentinel (this binary)
 *       |
 *       v monitors /tmp/ringrift_watchdog.heartbeat
 *   master_loop_watchdog.py
 *       |
 *       v supervises
 *   master_loop.py
 *
 * The sentinel monitors the heartbeat file's mtime. If the file hasn't
 * been touched in STALE_THRESHOLD seconds, it kills any existing watchdog
 * and starts a fresh one.
 *
 * Build:
 *   cc -o ringrift_sentinel ringrift_sentinel.c
 *
 * Install (macOS):
 *   sudo cp ringrift_sentinel /usr/local/bin/
 *   sudo launchctl load /Library/LaunchDaemons/com.ringrift.sentinel.plist
 *
 * January 2026: Created for cluster resilience (Session 16).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <syslog.h>

/* Configuration - can be overridden via environment */
#define DEFAULT_HEARTBEAT_PATH "/tmp/ringrift_watchdog.heartbeat"
#define DEFAULT_CHECK_INTERVAL 30      /* seconds between checks */
#define DEFAULT_STALE_THRESHOLD 120    /* seconds until heartbeat is stale */
#define DEFAULT_STARTUP_GRACE 60       /* seconds to wait after restart */

/* Paths to watchdog - try these in order */
static const char *WATCHDOG_PATHS[] = {
    "/Users/armand/Development/RingRift/ai-service/scripts/master_loop_watchdog.py",
    "./scripts/master_loop_watchdog.py",
    "/opt/ringrift/ai-service/scripts/master_loop_watchdog.py",
    NULL
};

/* Python paths to try */
static const char *PYTHON_PATHS[] = {
    "/Users/armand/.pyenv/versions/3.10.13/bin/python",
    "/usr/local/bin/python3",
    "/usr/bin/python3",
    "python3",
    NULL
};

/* Global state */
static volatile int running = 1;
static time_t last_restart = 0;
static int restart_count = 0;

/* Configuration (loaded from environment) */
static const char *heartbeat_path;
static int check_interval;
static int stale_threshold;
static int startup_grace;

/*
 * Signal handler for graceful shutdown.
 */
static void handle_signal(int sig) {
    syslog(LOG_INFO, "Received signal %d, shutting down", sig);
    running = 0;
}

/*
 * Get file modification time.
 * Returns -1 on error, otherwise the mtime as time_t.
 */
static time_t get_file_mtime(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    return st.st_mtime;
}

/*
 * Check if heartbeat file is stale.
 * Returns 1 if stale or missing, 0 if fresh.
 */
static int is_heartbeat_stale(void) {
    time_t mtime = get_file_mtime(heartbeat_path);
    if (mtime == -1) {
        syslog(LOG_WARNING, "Heartbeat file not found: %s", heartbeat_path);
        return 1;
    }

    time_t now = time(NULL);
    int age = (int)(now - mtime);

    if (age > stale_threshold) {
        syslog(LOG_WARNING, "Heartbeat stale: %d seconds old (threshold: %d)",
               age, stale_threshold);
        return 1;
    }

    syslog(LOG_DEBUG, "Heartbeat fresh: %d seconds old", age);
    return 0;
}

/*
 * Find the watchdog script path.
 * Returns NULL if not found.
 */
static const char *find_watchdog_path(void) {
    for (int i = 0; WATCHDOG_PATHS[i] != NULL; i++) {
        if (access(WATCHDOG_PATHS[i], F_OK) == 0) {
            return WATCHDOG_PATHS[i];
        }
    }
    return NULL;
}

/*
 * Find Python interpreter.
 * Returns NULL if not found.
 */
static const char *find_python_path(void) {
    for (int i = 0; PYTHON_PATHS[i] != NULL; i++) {
        if (access(PYTHON_PATHS[i], X_OK) == 0) {
            return PYTHON_PATHS[i];
        }
    }
    return NULL;
}

/*
 * Kill any existing watchdog processes.
 */
static void kill_existing_watchdog(void) {
    /* Use pkill to find and kill watchdog processes */
    int ret = system("pkill -f 'master_loop_watchdog\\.py' 2>/dev/null");
    if (ret == 0) {
        syslog(LOG_INFO, "Killed existing watchdog process(es)");
        sleep(2);  /* Give time for cleanup */
    }
}

/*
 * Start the watchdog process.
 * Returns 0 on success, -1 on failure.
 */
static int start_watchdog(void) {
    const char *watchdog = find_watchdog_path();
    const char *python = find_python_path();

    if (watchdog == NULL) {
        syslog(LOG_ERR, "Cannot find watchdog script");
        return -1;
    }

    if (python == NULL) {
        syslog(LOG_ERR, "Cannot find Python interpreter");
        return -1;
    }

    syslog(LOG_INFO, "Starting watchdog: %s %s", python, watchdog);

    /* Fork and exec */
    pid_t pid = fork();
    if (pid < 0) {
        syslog(LOG_ERR, "Fork failed: %s", strerror(errno));
        return -1;
    }

    if (pid == 0) {
        /* Child process */

        /* Create new session (detach from terminal) */
        setsid();

        /* Close stdin and redirect stdout/stderr to log */
        close(STDIN_FILENO);

        /* Set up environment */
        setenv("PYTHONPATH",
               "/Users/armand/Development/RingRift/ai-service",
               1);

        /* Exec the watchdog */
        execlp(python, python, watchdog, NULL);

        /* If we get here, exec failed */
        syslog(LOG_ERR, "Exec failed: %s", strerror(errno));
        _exit(1);
    }

    /* Parent process */
    syslog(LOG_INFO, "Watchdog started with PID %d", pid);
    last_restart = time(NULL);
    restart_count++;

    return 0;
}

/*
 * Check if we should restart (respecting grace period).
 */
static int can_restart(void) {
    if (last_restart == 0) {
        return 1;  /* Never restarted before */
    }

    time_t now = time(NULL);
    int elapsed = (int)(now - last_restart);

    if (elapsed < startup_grace) {
        syslog(LOG_DEBUG, "In grace period (%d/%d seconds), skipping restart",
               elapsed, startup_grace);
        return 0;
    }

    return 1;
}

/*
 * Load configuration from environment.
 */
static void load_config(void) {
    const char *env;

    /* Heartbeat path */
    env = getenv("RINGRIFT_SENTINEL_HEARTBEAT_PATH");
    heartbeat_path = env ? env : DEFAULT_HEARTBEAT_PATH;

    /* Check interval */
    env = getenv("RINGRIFT_SENTINEL_CHECK_INTERVAL");
    check_interval = env ? atoi(env) : DEFAULT_CHECK_INTERVAL;

    /* Stale threshold */
    env = getenv("RINGRIFT_SENTINEL_STALE_THRESHOLD");
    stale_threshold = env ? atoi(env) : DEFAULT_STALE_THRESHOLD;

    /* Startup grace */
    env = getenv("RINGRIFT_SENTINEL_STARTUP_GRACE");
    startup_grace = env ? atoi(env) : DEFAULT_STARTUP_GRACE;

    syslog(LOG_INFO, "Config: heartbeat=%s, interval=%d, threshold=%d, grace=%d",
           heartbeat_path, check_interval, stale_threshold, startup_grace);
}

/*
 * Main entry point.
 */
int main(int argc, char *argv[]) {
    /* Open syslog */
    openlog("ringrift_sentinel", LOG_PID | LOG_NDELAY, LOG_DAEMON);
    syslog(LOG_INFO, "RingRift Sentinel starting");

    /* Load configuration */
    load_config();

    /* Set up signal handlers */
    signal(SIGTERM, handle_signal);
    signal(SIGINT, handle_signal);
    signal(SIGHUP, SIG_IGN);  /* Ignore hangup */

    /* Initial delay to allow system startup */
    syslog(LOG_INFO, "Initial grace period: %d seconds", startup_grace);
    sleep(startup_grace);

    /* Main loop */
    while (running) {
        if (is_heartbeat_stale()) {
            if (can_restart()) {
                syslog(LOG_WARNING, "Heartbeat stale, restarting watchdog "
                       "(restart #%d)", restart_count + 1);
                kill_existing_watchdog();
                if (start_watchdog() == 0) {
                    syslog(LOG_INFO, "Watchdog restart successful");
                } else {
                    syslog(LOG_ERR, "Watchdog restart failed");
                }
            }
        }

        sleep(check_interval);
    }

    syslog(LOG_INFO, "RingRift Sentinel shutting down (restarts: %d)",
           restart_count);
    closelog();

    return 0;
}
