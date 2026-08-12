#include "qemu_io.h"

static uint32_t sh_call(uint32_t op, void *arg)
{
    register uint32_t r0 __asm__("r0") = op;
    register void    *r1 __asm__("r1") = arg;
    __asm__ volatile ("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}
void sh_write0(const char *s) { sh_call(0x04u, (void *)(uintptr_t)s); }
void sh_exit(void)
{
    uint32_t a[2] = { 0x20026u, 0u };
    sh_call(0x18u, a);
}
static uint32_t slen(const char *s) { uint32_t n = 0; while (s[n]) ++n; return n; }
int sh_open_wb(const char *name)
{
    uint32_t a[3] = { (uint32_t)(uintptr_t)name, 5u /* "wb" */, slen(name) };
    return (int)sh_call(0x01u, a);
}
int sh_write(int fd, const void *p, uint32_t n)
{
    uint32_t a[3] = { (uint32_t)fd, (uint32_t)(uintptr_t)p, n };
    return (int)sh_call(0x05u, a);      /* 0 == everything written */
}
void sh_close(int fd) { uint32_t a[1] = { (uint32_t)fd }; sh_call(0x02u, a); }

static char *utoa_(uint32_t v, char *p)
{
    char t[12]; int n = 0;
    do { t[n++] = (char)('0' + v % 10u); v /= 10u; } while (v);
    while (n) *p++ = t[--n];
    return p;
}
void say(const char *label, uint32_t v)
{
    char buf[96], *p = buf;
    while (*label) *p++ = *label++;
    p = utoa_(v, p);
    *p++ = '\n'; *p = 0;
    sh_write0(buf);
}

/* newlib stubs: malloc is in npu_stub, the rest is never called */
extern uint32_t __heap_start__, __heap_end__;
void *_sbrk(int incr)
{
    static char *cur;
    if (!cur) cur = (char *)&__heap_start__;
    char *prev = cur;
    if (cur + incr > (char *)&__heap_end__) return (void *)-1;
    cur += incr;
    return prev;
}
int _write(int f, char *p, int n) { (void)f; (void)p; return n; }
int _close(int f) { (void)f; return -1; }
int _fstat(int f, void *st) { (void)f; (void)st; return 0; }
int _isatty(int f) { (void)f; return 1; }
int _lseek(int f, int o, int w) { (void)f; (void)o; (void)w; return 0; }
int _read(int f, char *p, int n) { (void)f; (void)p; (void)n; return 0; }
void _exit(int c) { (void)c; for (;;) {} }
int _kill(int p, int s) { (void)p; (void)s; return -1; }
int _getpid(void) { return 1; }
