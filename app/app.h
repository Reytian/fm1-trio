/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO: play the chords; a bass player and a drummer, both small neural networks, walk with you.
 * This header is the app's half of the contract with fm1-base (device/unity.c, host/plat_host.c).
 * (Started from fm1-base's template, by Charles Vestal.) */
#pragma once
#include <stdint.h>
#include "plat.h"

#define APP_NAME "TRIO"
#define APP_SPLASH_RGB RGB(232, 160, 72)
#define APP_BLOCK 256u                   /* frames per audio block: 5.8 ms at 44.1 kHz */
#define APP_TOUCH                        /* velocity from the keys (lib/touch.h): the piano plays it */
#define APP_BTDEV                        /* Bluetooth MIDI devices, a keyboard or a piano (lib/btdev.h): SEL's page */
#define APP_BLE_AT_BOOT (!trio_ble_off())   /* Bluetooth at power-on: as the player saved it (SEL's page, PRESET) */
int trio_ble_off(void);

/* the store: the settings, the panel's calibration (the platform's), Bluetooth's address and pairing, and what the
 * bass learned from the player (EDIT: the BASS network's output rows, 3,760 bytes) at 0xDA000, two sectors below the
 * calibration (the data area's 0x97000..0xD9FFF stays free). An object's number is in its record: keep them (a new
 * object goes last) */
enum { OBJ_SETTINGS, OBJ_PANEL, OBJ_BLE, OBJ_BASS, OBJ_NOBJ };
#define APP_STORE_MAP {0xFC000u, 0xDC000u, 0xDE000u, 0xDA000u}

/* the contract: the platform calls these */
void app_init(void);                     /* boot: the engine, then the saved settings (or defaults) */
void app_render(int32_t *lr, uint32_t n, uint32_t master_q12);   /* audio interrupt: n stereo frames, 24-bit */
void app_panic(void);                    /* silence now (before an update rewrites the flash) */
void app_ui_init(void);                  /* the first screen, after the splash */
void app_frame(void);                    /* the UI: input and a redraw, about 60 times a second */
void app_input_only(void);               /* between frames: input only */
void app_say(const char *a, const char *b);   /* a short message on the screen and to a screen reader */
#define APP_LOOP
void app_loop(void);                     /* every pass of the main loop: the networks (dsp/trio.c tr_service) */

/* the host simulator's checks (host/plat_host.c "expect"), and its reboot (what the app's RAM forgets) */
#define APP_HOST_EXPECT
int app_expect(const char *what, int *got);
#define APP_HOST_REBOOT
void app_host_reboot(void);
