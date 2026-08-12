/* qemu_io.h — output and files via ARM semihosting (BKPT 0xAB), without
 * newlib. For host runs of the firmware code in QEMU only. */
#ifndef N6_QEMU_IO_H
#define N6_QEMU_IO_H
#include <stdint.h>
void     sh_write0(const char *s);
void     say(const char *label, uint32_t v);
int      sh_open_wb(const char *name);
int      sh_write(int fd, const void *p, uint32_t n);
void     sh_close(int fd);
void     sh_exit(void);
#endif
