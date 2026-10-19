/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO on fm1-base: the screen, the controls, the settings, and the glue to the engine (dsp/trio.c).
 * Part of fm1-base's unity build (device/unity.c, host/plat_host.c), after the platform: gfx.c, announce.c and
 * touch.c are in scope. (Its shape is fm1-base's template's, by Charles Vestal.)
 *
 *   the keys, MIDI in   the player's chords (and the piano's sound, unless an outside piano makes its own); MIDI
 *                       over USB, TRS or Bluetooth (a computer connected to the FM-1, or a keyboard it connected to)
 *   PLAY (or SEQ)       the band: a bar's count-in, then it walks; again: it stops
 *   K1 TEMPO  K2 FREEDOM (the bass)  K3 DRUMS (the drummer's energy)  K4 SWING
 *   ARP                 the feel: AUTO (in two until the form comes round), TWO, FOUR
 *   REC                 forget the form (a new tune)
 *   GLO                 the mix page: K1 PIANO  K2 BASS  K3 DRUMS  K4 ROOM; PRESET: the piano's sound on or off
 *   SEL                 the Bluetooth page: ALGO picks a MIDI device (a keyboard, a piano), SEL connects to it or
 *                       lets it go; PRESET: Bluetooth on or off; GLO: back to the band
 *   EDIT                teach the bass: TRIO's bass quiet, your notes from E1 to G3 the bass; EDIT again (or STOP):
 *                       it learns your line. Held 1.5 s: the bass as trained
 *   OCT- / OCT+         the keys an octave down or up; SAVE keeps the settings and what the bass learned */
#include "fastmath.h"
#include "trio.h"

#define SET_MAGIC 0x54524931u            /* "TRI1": the settings record; another magic loads defaults */
typedef struct {
    uint32_t magic;
    tr_set_t tr;
    int8_t octave;                       /* the keys: -2 .. +1 */
    uint8_t ble_off;                     /* Bluetooth switched off (SEL's page, PRESET); 0, the default: on (0.1's
                                          * records have 0 here: it was reserved, the size is the same) */
    uint8_t rsv[2];
} settings_t;
static settings_t set;
static uint8_t set_dirty;                /* changed since the last save: SAVE is lit */
static uint32_t bass_saved_at;           /* the engine's count of learnings and forgettings when the bass was saved or
                                          * loaded: past it, what it learned is unsaved (SAVE lit) */
#define EDIT_FORGET_MS 1500u             /* EDIT held this long: the bass as trained */

static const char *const FEEL_NAME[3] = {"AUTO", "TWO", "FOUR"};

/* ------------------------------------------------------------------ the engine's glue --- */
/* the notes held over Bluetooth, and its sustain pedal: let go if the link ends mid-chord (a piano or a computer
 * gone, let go, Bluetooth off), else the piano would hold them and the listening would hear them in every chord
 * after. The audio interrupt keeps them; the main loop asks (ble_release) when a link ends */
static uint32_t ble_held[4];
static uint8_t ble_pedal;
static volatile uint8_t ble_release;

void app_render(int32_t *lr, uint32_t n, uint32_t master_q12)
{
    uint32_t pkt, k;
    int port;
    if (ble_release) {
        ble_release = 0;
        for (k = 0; k < 128u; k++)
            if (ble_held[k >> 5] >> (k & 31u) & 1u)
                tr_midi(0x80u, k, 0);
        if (ble_pedal)
            tr_midi(0xB0u, 64u, 0);
        ble_held[0] = ble_held[1] = ble_held[2] = ble_held[3] = 0;
        ble_pedal = 0;
    }
    while (plat_midi_in(&pkt, &port)) {  /* a keyboard over USB, TRS or Bluetooth, any channel */
        uint32_t st = (pkt >> 8) & 0xFFu, d1 = (pkt >> 16) & 0x7Fu, d2 = (pkt >> 24) & 0x7Fu, type = st & 0xF0u;
        if (st < 0x80u || st >= 0xF0u)
            continue;
        if (port == PORT_BLE) {
            if (type == 0x90u && d2)
                ble_held[d1 >> 5] |= 1u << (d1 & 31u);
            else if (type == 0x80u || type == 0x90u)
                ble_held[d1 >> 5] &= ~(1u << (d1 & 31u));
            else if (type == 0xB0u && d1 == 64u)
                ble_pedal = d2 >= 64u;
        }
        tr_midi(st, d1, d2);
    }
    tr_render(lr, n, master_q12);
}

void app_loop(void)                      /* the networks: one step a pass (a few ms at most), so the keys and the
                                          * screen never wait long; the bass decides 40 ms ahead, the drums 6 steps */
{
    tr_service();
}

void app_panic(void) { tr_panic(); }

/* --------------------------------------------------------------- settings --- */
static void settings_defaults(void)
{
    set.magic = SET_MAGIC;
    tr_defaults(&set.tr);
    set.octave = 0;
    set.ble_off = 0;
}

static int settings_ok(void)
{
    const tr_set_t *t = &set.tr;
    return set.magic == SET_MAGIC && t->bpm >= 60u && t->bpm <= 260u && t->feel <= TR_FOUR && t->freedom <= 100u &&
           t->energy <= 100u && t->swing >= 50u && t->swing <= 75u && t->piano <= 1u && t->vol_piano <= 100u &&
           t->vol_bass <= 100u && t->vol_drums <= 100u && t->room <= 100u && t->follow <= 1u && set.octave >= -2 &&
           set.octave <= 1 && set.ble_off <= 1u;
}

/* what the bass learned (EDIT): its own object, in the form the engine keeps it in (tr_learned_obj), so it goes to the
 * flash and back as it is; none, damaged or another network's: the bass as trained */
static int bass_unsaved(void) { return tr_learn_count != bass_saved_at; }

static void bass_load(void)
{
    uint32_t max;
    void *p = tr_learned_room(&max);
    tr_learned_check(plat_store_load(OBJ_BASS, p, max));
    bass_saved_at = tr_learn_count;
}

static int bass_save(void)
{
    uint32_t n, at = tr_learn_count;
    const void *p = tr_learned_obj(&n);
    int rc = plat_store_save(OBJ_BASS, p, n);
    if (rc == 0)
        bass_saved_at = at;
    return rc;
}

void app_init(void)
{
    tr_init();
    if (plat_store_load(OBJ_SETTINGS, &set, sizeof set) != (int)sizeof set || !settings_ok())
        settings_defaults();
    tr_set = set.tr;
    bass_load();
}

int trio_ble_off(void) { return set.ble_off; }   /* (app.h APP_BLE_AT_BOOT: at power-on, after app_init) */

static void apply(void)                  /* a setting changed: into the engine, and SAVE lit */
{
    tr_set = set.tr;
    set_dirty = 1;
}

/* --------------------------------------------------------------------- UI --- */
enum { PG_BAND, PG_MIX, PG_BT };         /* the pages: the band (K1 .. K4), the mix (GLO), Bluetooth (SEL) */
static struct {
    char msg[2][24];                     /* a short message over the header */
    uint32_t msg_until;
    uint32_t keys, btn;                  /* held at the last input */
    uint8_t page;                        /* PG_* */
    uint8_t redraw, header;              /* the header and the controls' strips to draw again; every strip */
    int turned;                          /* the knob turned last (EN_*): lit a moment */
    uint32_t turned_at;
    int32_t shown[7];                    /* what the screen shows: a change draws its strip again */
    uint8_t radio_up;                    /* Bluetooth's radio started since power-on (switched on again: no restart) */
    uint8_t links;                       /* Bluetooth's links at the last frame: 1 a computer, 2 a device */
    int found;                           /* devices found while the Bluetooth page is up, as last named */
    uint32_t edit_at;                    /* EDIT went down (0: up, or its hold has acted) */
    uint8_t learning;                    /* the engine was learning a line at the last frame */
} ui;

static void text_cat(char *d, const char *s)
{
    while (*d)
        d++;
    while (*s)
        *d++ = *s++;
    *d = 0;
}

static void int_text(int32_t v, char *b)     /* signed, no padding */
{
    char t[12];
    int n = 0;
    uint32_t u = (uint32_t)(v < 0 ? -v : v);
    do
        t[n++] = (char)('0' + u % 10u);
    while ((u /= 10u) && n < 11);
    if (v < 0)
        *b++ = '-';
    while (n)
        *b++ = t[--n];
    *b = 0;
}

void app_say(const char *a, const char *b)
{
    uint32_t i;
    char t[48];
    for (i = 0; i < 23u && a && a[i]; i++)
        ui.msg[0][i] = a[i];
    ui.msg[0][i] = 0;
    for (i = 0; i < 23u && b && b[i]; i++)
        ui.msg[1][i] = b[i];
    ui.msg[1][i] = 0;
    ui.msg_until = plat_ms() + 1300u;
    ui.redraw = 1;
    if (ann_on()) {                          /* and to a screen reader (announce.c, fm1-reader) */
        t[0] = 0;
        text_cat(t, ui.msg[0]);
        if (ui.msg[1][0]) {
            text_cat(t, " ");
            text_cat(t, ui.msg[1]);
        }
        ann_say(t, 0, ANN_NOW);
    }
}

static void say_value(const char *name, int32_t v, const char *unit)
{
    char t[16];
    int_text(v, t);
    text_cat(t, unit);
    app_say(name, t);
}

static void bar_beat(int32_t b, char *t)     /* "12.3"; the count-in: "4" .. "1" */
{
    char u[8];
    t[0] = 0;
    if (b < 0) {
        int_text(-b, t);
        return;
    }
    int_text(b / 4 + 1, t);
    text_cat(t, ".");
    int_text(b % 4 + 1, u);
    text_cat(t, u);
}

static void note_text(int note, char *b)
{
    static const char *const N[12] = {"C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"};
    char o[8];
    b[0] = 0;
    if (note < 0) {
        text_cat(b, "-");
        return;
    }
    text_cat(b, N[note % 12]);
    int_text(note / 12 - 1, o);
    text_cat(b, o);
}

/* -------------------------------------------------------------- Bluetooth --- */
/* MIDI in from a computer or a tablet connected to the FM-1, and from a keyboard or a piano the FM-1 connects to
 * (lib/btdev: the scan while SEL's page is up, ALGO's pick, SEL's connect and let go) */
static uint8_t let_go;                   /* SEL let a device go: its end is not a NO ANSWER */

static int ble_on(void) { return FM1B_BLE && !set.ble_off && plat_ble_state() != BLE_OFF; }

static void device_say(void)             /* the device ALGO picks, as it is found or picked */
{
    char name[21];
    if (btdev_found(btdev_pick(), name, 0))
        app_say("DEVICE", name);
}

/* SEL's page, PRESET: Bluetooth off or on, kept with the settings (as it is at the next power-on). A radio that has
 * not started since power-on starts now, and the FM-1 holds its sound for the radio's first second (device/radio.c):
 * the band would stop for it, so then only while the band is stopped */
static void ble_switch(void)
{
    int on = !ble_on();
    if (on && !ui.radio_up && tr_state != TR_STOP) {
        app_say("BLUETOOTH", "STOP FIRST");
        return;
    }
    set.ble_off = (uint8_t)!on;
    set_dirty = 1;
    plat_ble_enable(on);
    app_say("BLUETOOTH", on ? "ON" : "OFF");
}

/* SEL: the Bluetooth page; on it, connect to the device ALGO picked, or let the one connected go */
static void sel_button(void)
{
    int ev;
    if (ui.page != PG_BT) {
        ui.page = PG_BT;
        app_say("BLUETOOTH", ble_on() ? "ON" : "OFF");
    } else if (!ble_on()) {
        app_say("BLUETOOTH", "OFF");
    } else {
        ev = btdev_save();
        if (ev == BTDEV_LET_GO)
            let_go = 1;
        app_say(btdev_word(ev), ev == BTDEV_NO_DEVICE ? "" : btdev_name());
    }
}

/* every frame: the scan while the page is up, what happened (in words), and a link's notes let go when it ends */
static void ble_frame(void)
{
    int n, ev, links;
    if (!FM1B_BLE)
        return;
    if (plat_ble_state() != BLE_OFF)
        ui.radio_up = 1;
    btdev_page(ui.page == PG_BT && !set.ble_off);
    n = ui.page == PG_BT ? btdev_count() : 0;
    if (n > ui.found && btdev_pick() < n)    /* a device found: named at once (SEL connects to it) */
        device_say();
    ui.found = n;
    while ((ev = btdev_event()) != BTDEV_NONE) {
        if (set.ble_off || (ev == BTDEV_NO_ANSWER && let_go))   /* switched off, or let go: said already */
            continue;
        if (ev == BTDEV_CONNECTED || ev == BTDEV_NO_ANSWER || ev == BTDEV_LOST)
            app_say(btdev_word(ev), btdev_name());
        else if (ev == BTDEV_COMPUTER || ev == BTDEV_COMPUTER_GONE)
            app_say("COMPUTER", ev == BTDEV_COMPUTER ? "CONNECTED" : "DISCONNECTED");
    }
    if (!btdev_busy())                       /* (the page may be scanning again already) */
        let_go = 0;
    links = plat_ble_connected() | (plat_ble_device() == BLE_DEV_READY) << 1;
    if (ui.links & ~links)                   /* a link ended: its notes still held let go (app_render) */
        ble_release = 1;
    ui.links = (uint8_t)links;
}

/* a screen reader that has just started listening (or asks) hears the whole screen */
static void speak(void)
{
    char t[192], u[16];
    int r, q, f;
    if (ann_on() && ann_where()) {
        t[0] = 0;
        text_cat(t, APP_NAME ". ");
        text_cat(t, tr_state == TR_STOP ? "STOPPED" : tr_state == TR_COUNT ? "COUNT IN" : "PLAYING");
        if (tr_chord_now(&r, &q)) {
            text_cat(t, ". CHORD ");
            tr_chord_name(r, q, u);
            text_cat(t, u);
        }
        text_cat(t, ". TEMPO ");
        int_text(set.tr.bpm, u);
        text_cat(t, u);
        text_cat(t, ". FEEL ");
        text_cat(t, FEEL_NAME[set.tr.feel]);
        if ((f = tr_form_bars()) != 0) {
            text_cat(t, ". FORM ");
            int_text(f, u);
            text_cat(t, u);
            text_cat(t, " BARS");
        }
        if (tr_teaching()) {
            text_cat(t, ". TEACHING THE BASS, ");
            int_text(tr_taught(), u);
            text_cat(t, u);
            text_cat(t, " BEATS");
        } else if (tr_learning()) {
            text_cat(t, ". LEARNING YOUR LINE");
        } else if (tr_learned()) {
            text_cat(t, ". BASS TAUGHT");
        }
        if (FM1B_BLE && !ble_on()) {
            text_cat(t, ". BLUETOOTH OFF");
        } else if (FM1B_BLE) {
            if (btdev_ready()) {
                text_cat(t, ". DEVICE ");
                text_cat(t, btdev_name());
            }
            if (plat_ble_connected())
                text_cat(t, ". COMPUTER CONNECTED");
        }
        ann_say(t, 0, ANN_NOW);
    }
    ann_tick(plat_ms());
}

/* ------------------------------------------------------------------ input --- */
static void knob(int k, int32_t d)
{
    tr_set_t *t = &set.tr;
    if (ui.page == PG_BT) {                  /* ALGO picks a device (while nothing is connected), PRESET switches */
        if (k == EN_ALGO && !btdev_busy() && btdev_count()) {
            btdev_choose(btdev_pick() + (int)d);
            device_say();
        } else if (k == EN_PRESET) {
            ble_switch();
        }
        return;
    }
    if (ui.page == PG_BAND) {
        switch (k) {
        case EN_K1:
            t->bpm = (uint16_t)clamp(t->bpm + d, 60, 260);
            say_value("TEMPO", t->bpm, " BPM");
            break;
        case EN_K2:
            t->freedom = (uint8_t)clamp(t->freedom + 2 * d, 0, 100);
            say_value("FREEDOM", t->freedom, "");
            break;
        case EN_K3:
            t->energy = (uint8_t)clamp(t->energy + 2 * d, 0, 100);
            say_value("DRUMS", t->energy, "");
            break;
        case EN_K4:
            t->swing = (uint8_t)clamp(t->swing + d, 50, 75);
            say_value("SWING", t->swing, "%");
            break;
        default:
            return;
        }
    } else {
        switch (k) {
        case EN_K1:
            t->vol_piano = (uint8_t)clamp(t->vol_piano + 2 * d, 0, 100);
            say_value("PIANO", t->vol_piano, "");
            break;
        case EN_K2:
            t->vol_bass = (uint8_t)clamp(t->vol_bass + 2 * d, 0, 100);
            say_value("BASS", t->vol_bass, "");
            break;
        case EN_K3:
            t->vol_drums = (uint8_t)clamp(t->vol_drums + 2 * d, 0, 100);
            say_value("DRUMS", t->vol_drums, "");
            break;
        case EN_K4:
            t->room = (uint8_t)clamp(t->room + 2 * d, 0, 100);
            say_value("ROOM", t->room, "");
            break;
        case EN_PRESET:
            t->piano = (uint8_t)!t->piano;
            app_say("PIANO SOUND", t->piano ? "ON" : "OFF");
            break;
        default:
            return;
        }
    }
    ui.turned = k;
    ui.turned_at = plat_ms();
    apply();
}

static void octave_step(int32_t d)
{
    int32_t o = clamp(set.octave + d, -2, 1);
    uint32_t k;
    char t[4];
    if (o == set.octave)
        return;
    for (k = 0; k < NKEYS; k++)              /* the keys held now let go of their notes (their release is the new
                                              * octave's: no note left hanging) */
        if (ui.keys >> k & 1u)
            tr_key(53 + (int)k + 12 * set.octave, 0);
    set.octave = (int8_t)o;
    set_dirty = 1;
    t[0] = o < 0 ? '-' : '+';
    t[1] = (char)('0' + (o < 0 ? -o : o));
    t[2] = 0;
    app_say("OCTAVE", t);
}

/* teaching off (EDIT, STOP): the line learned, if it had notes enough */
static void teach_off(void)
{
    int n = tr_teach(0);
    char t[16];
    if (!n) {
        app_say("NOTHING LEARNED", "PLAY A LINE");
        return;
    }
    int_text((n + 2) / 4, t);
    text_cat(t, " BARS");
    app_say("LEARNING", t);
}

/* EDIT let go before its hold acted: teaching on, or off (the line learned) */
static void edit_tap(void)
{
    if (tr_teaching())
        teach_off();
    else if (tr_learning())
        app_say("STILL LEARNING", "");
    else if (tr_teach(1))
        app_say("TEACH BASS", tr_state == TR_STOP ? "PRESS PLAY" : "PLAY THE LINE");
}

/* EDIT held: the bass as trained (a teaching, a learning under way dropped too) */
static void edit_hold(void)
{
    tr_unlearn();
    ui.learning = 0;                         /* (not a learning come to its end: nothing LEARNED to say) */
    app_say("BASS", "AS TRAINED");
}

static void button(uint32_t k)
{
    switch (k) {
    case B_PLAY:
    case B_SEQ:
        if (tr_state == TR_STOP) {
            tr_play(1);
            app_say(tr_teaching() ? "TEACH BASS" : "COUNT IN", tr_teaching() ? "COUNT IN" : "");
        } else if (tr_teaching()) {          /* STOP while teaching: it learns the line */
            teach_off();
            tr_play(0);
        } else {
            tr_play(0);
            app_say("STOPPED", "");
        }
        break;
    case B_EDIT:                             /* (decided when it is let go, or held: edit_tap, edit_hold) */
        ui.edit_at = plat_ms() ? plat_ms() : 1u;
        break;
    case B_ARP:
        set.tr.feel = (uint8_t)((set.tr.feel + 1u) % 3u);
        apply();
        app_say("FEEL", FEEL_NAME[set.tr.feel]);
        break;
    case B_REC:
        tr_forget();
        app_say("FORM", "FORGOTTEN");
        break;
    case B_GLO:                              /* the mix page and back; from the Bluetooth page, the band's */
        ui.page = ui.page == PG_BAND ? PG_MIX : PG_BAND;
        app_say(ui.page == PG_MIX ? "MIX" : "BAND", "");
        break;
    case B_SEL:
        if (FM1B_BLE)
            sel_button();
        break;
    case B_OCTDN:
        octave_step(-1);
        break;
    case B_OCTUP:
        octave_step(1);
        break;
    case B_SAVE:                             /* the settings, and what the bass learned if that changed */
        if (plat_store_save(OBJ_SETTINGS, &set, sizeof set) == 0 && (!bass_unsaved() || bass_save() == 0)) {
            set_dirty = 0;
            app_say("SAVED", "");
        } else {
            app_say("SAVE", "FAILED");
        }
        break;
    default:
        break;
    }
}

static void input(void)
{
    uint32_t keys, btn = plat_buttons(), k;
    int32_t e;
    touch_poll();                            /* the sensed keys' presses and velocities */
    keys = touch_keys();
    for (k = 0; k < NKEYS; k++) {            /* key 0 is F3 (MIDI 53) */
        uint32_t bit = 1u << k;
        int note = 53 + (int)k + 12 * set.octave;
        if ((keys & bit) && !(ui.keys & bit))
            tr_key(note, (int)touch_vel(k));
        else if (!(keys & bit) && (ui.keys & bit))
            tr_key(note, 0);
    }
    ui.keys = keys;                          /* (the keys are not drawn: playing them redraws nothing) */
    for (k = 0; k < NB; k++)
        if ((btn >> k & 1u) && !(ui.btn >> k & 1u))
            button(k);
    if (ui.edit_at && !(btn >> B_EDIT & 1u)) {   /* EDIT let go: a tap */
        ui.edit_at = 0;
        edit_tap();
    } else if (ui.edit_at && plat_ms() - ui.edit_at >= EDIT_FORGET_MS) {   /* held: forget */
        ui.edit_at = 0;
        edit_hold();
    }
    ui.btn = btn;
    for (k = 0; k < NE; k++)
        if ((e = plat_enc((int)k)) != 0)
            knob((int)k, e);
    plat_glow(1);                            /* the unlit buttons glow, readable in the dark */
    plat_leds(btn | (tr_state != TR_STOP ? 1u << B_PLAY : 0u) | (set_dirty || bass_unsaved() ? 1u << B_SAVE : 0u) |
                  (ui.page == PG_MIX ? 1u << B_GLO : 0u) | (ui.page == PG_BT ? 1u << B_SEL : 0u) |
                  (tr_teaching() || tr_learning() ? 1u << B_EDIT : 0u),
              keys);
}

void app_input_only(void) { input(); }

/* ---------------------------------------------------------------- drawing --- */
static void knob_cell(int32_t x, const char *name, int32_t v, const char *unit, int lit)
{
    char t[12];
    cv_text(x, 0, &FONT_XS, name, lit ? C_HI : C_GRAY);
    int_text(v, t);
    text_cat(t, unit);
    cv_text(x, 14, &FONT_B, t, lit ? C_WHITE : C_HI);
}

static void draw_header(void)            /* the name, the band's state and where it is (bar.beat), or a message */
{
    char t[24];
    cv_begin(240, 32, C_BLACK);
    if (plat_ms() < ui.msg_until) {
        int32_t x = cv_text(10, 6, &FONT_B, ui.msg[0], C_WHITE);
        cv_text(x + 8, 6, &FONT_B, ui.msg[1], C_HI);
    } else {
        int32_t x = cv_text(10, 6, &FONT_B, APP_NAME, C_WHITE);
        if (ble_on()) {                      /* Bluetooth: BLE lit while a computer is connected, DEV while a keyboard
                                              * or a piano is */
            x = cv_text(x + 8, 10, &FONT_XS, "BLE", plat_ble_connected() ? C_HI : C_DIM);
            if (FM1B_BLE_CENTRAL)
                x = cv_text(x + 6, 10, &FONT_XS, "DEV", plat_ble_device() == BLE_DEV_READY ? C_HI : C_DIM);
        }
        x = cv_text(x + 10, 10, &FONT_XS,
                    tr_state == TR_STOP ? "STOPPED" : tr_state == TR_COUNT ? "COUNT IN" : tr_teaching() ? "TEACHING"
                                                                                                   : "PLAYING",
                    tr_state != TR_PLAY ? C_GRAY : tr_teaching() ? C_AMB : C_HI);
        if (tr_missed) {                     /* beats the networks were late for (should never show) */
            char u[16] = "LATE ";
            int_text((int32_t)tr_missed, u + 5);
            cv_text(x + 8, 10, &FONT_XS, u, C_DIM);
        }
        if (tr_state != TR_STOP) {
            bar_beat(tr_beat, t);
            cv_text(230 - text_w(&FONT_B, t), 6, &FONT_B, t, C_HI);
        }
    }
    cv_rect(0, 31, 240, 1, C_LINE);
    cv_blit(0, 0);
}

static void draw_chord(void)             /* the chord the band plays, big */
{
    char t[24];
    int root, qual;
    cv_begin(240, 48, C_BLACK);
    if (tr_chord_now(&root, &qual)) {
        tr_chord_name(root, qual, t);
        cv_text((240 - text_w(&FONT_L, t)) / 2, 10, &FONT_L, t, C_WHITE);
    } else {
        cv_text((240 - text_w(&FONT_B, "PLAY A CHORD")) / 2, 18, &FONT_B, "PLAY A CHORD", C_GRAY);
    }
    cv_blit(0, 34);
}

static void draw_info(void)              /* what comes next and from where, the form, the bass's note: each beat */
{
    char t[24], u[24];
    int nr, nq, nin, how, f;
    cv_begin(240, 48, C_BLACK);
    how = tr_chord_next(&nr, &nq, &nin);
    if (how) {                               /* 1: the form knows it; 2: the network's guess */
        const char *lab = how == 1 ? "NEXT" : "NEXT?";
        int32_t x;
        tr_chord_name(nr, nq, t);
        u[0] = 0;
        text_cat(u, "IN ");
        int_text(nin, u + 3);
        x = (240 - text_w(&FONT_XS, lab) - text_w(&FONT_B, t) - text_w(&FONT_XS, u) - 16) / 2;
        x = cv_text(x, 6, &FONT_XS, lab, C_GRAY);
        x = cv_text(x + 8, 2, &FONT_B, t, how == 1 ? C_HI : C_AMB);
        cv_text(x + 8, 6, &FONT_XS, u, C_GRAY);
    }
    f = tr_form_bars();
    u[0] = 0;
    if (tr_teaching()) {                     /* teaching: the bars of your line it keeps (32 at most: the last) */
        text_cat(u, "YOUR LINE ");
        int_text((tr_taught() + 2) / 4, t);
        text_cat(u, t);
        text_cat(u, " BARS");
    } else if (tr_learning()) {
        text_cat(u, "LEARNING YOUR LINE");
    } else if (f) {
        text_cat(u, "FORM ");
        int_text(f, t);
        text_cat(u, t);
        text_cat(u, " BARS");
    } else if (tr_state != TR_STOP) {
        text_cat(u, "LEARNING THE FORM");
    }
    cv_text(10, 28, &FONT_XS, u, tr_teaching() || tr_learning() ? C_AMB : f ? C_HI : C_DIM);
    if (tr_bass_note() >= 0) {               /* the bass's last note (teaching: yours); TAUGHT once it learned */
        note_text(tr_bass_note(), t);
        u[0] = 0;
        text_cat(u, "BASS ");
        text_cat(u, t);
        cv_text(230 - text_w(&FONT_XS, u), 28, &FONT_XS, u, C_GRAY);
        if (tr_learned() && !tr_teaching())
            cv_text(230 - text_w(&FONT_XS, u) - 8 - text_w(&FONT_XS, "TAUGHT"), 28, &FONT_XS, "TAUGHT", C_AMB);
    } else if (tr_learned()) {
        cv_text(230 - text_w(&FONT_XS, "BASS TAUGHT"), 28, &FONT_XS, "BASS TAUGHT", C_AMB);
    }
    cv_blit(0, 82);
}

static void draw_bt(void)                /* the Bluetooth page's half: a computer, Bluetooth on or off, the devices */
{
    static const char *const STATE[4] = {"BLUETOOTH OFF", "BLUETOOTH STARTING", "BLUETOOTH ON", "BLUETOOTH FAILED"};
    char t[24], u[8], name[21];
    int st = ble_on() ? plat_ble_state() & 3 : BLE_OFF, n = btdev_count(), comp = plat_ble_connected();
    cv_begin(240, 36, C_BLACK);
    cv_text(10, 4, &FONT_XS, comp ? "COMPUTER CONNECTED" : "NO COMPUTER", comp ? C_HI : C_DIM);
    cv_text(230 - text_w(&FONT_XS, STATE[st]), 4, &FONT_XS, STATE[st], st == BLE_OFF || st == BLE_FAILED ? C_HI : C_GRAY);
    cv_rect(0, 26, 240, 1, C_LINE);
    cv_blit(0, 132);
    cv_begin(240, 40, C_BLACK);              /* the device connected or connecting, or ALGO's pick of those found */
    if (st == BLE_OFF || st == BLE_FAILED) {
        cv_text(10, 0, &FONT_XS, st == BLE_OFF ? "Bluetooth is off" : "Bluetooth did not start", C_GRAY);
    } else if (btdev_busy()) {
        cv_text(10, 0, &FONT_XS, btdev_ready() ? "Connected" : "Connecting", C_GRAY);
        cv_text(10, 14, &FONT_B, btdev_name(), btdev_ready() ? C_WHITE : C_AMB);
    } else if (n && btdev_found(btdev_pick(), name, 0)) {
        cv_text(10, 0, &FONT_XS, "MIDI devices", C_GRAY);
        t[0] = 0;
        text_cat(t, "ALGO ");
        int_text(btdev_pick() + 1, u);
        text_cat(t, u);
        text_cat(t, "/");
        int_text(n, u);
        text_cat(t, u);
        cv_text(230 - text_w(&FONT_XS, t), 0, &FONT_XS, t, C_GRAY);
        cv_text(10, 14, &FONT_B, name, C_HI);
    } else {
        cv_text(10, 0, &FONT_XS, "Looking for MIDI devices", C_GRAY);
    }
    cv_blit(0, 172);
}

static void draw_rest(void)              /* the feel, the piano's sound, this page's knobs (or Bluetooth), the footer */
{
    uint32_t now = plat_ms();
    int lit = ui.turned_at && now - ui.turned_at < 600u;
    const tr_set_t *s = &set.tr;
    if (ui.page == PG_BT) {
        draw_bt();
    } else {
        cv_begin(240, 36, C_BLACK);
        {
            const char *pn = s->piano ? "PIANO ON" : "PIANO OFF";
            int32_t x = cv_text(10, 4, &FONT_XS, "FEEL", C_GRAY);
            x = cv_text(x + 6, 4, &FONT_XS, FEEL_NAME[s->feel], C_HI);
            if (s->feel == TR_AUTO && tr_state != TR_STOP)
                cv_text(x + 6, 4, &FONT_XS, tr_feel_now() == TR_TWO ? "(IN TWO)" : "(IN FOUR)", C_GRAY);
            cv_text(230 - text_w(&FONT_XS, pn), 4, &FONT_XS, pn, s->piano ? C_GRAY : C_HI);
            cv_rect(0, 26, 240, 1, C_LINE);
        }
        cv_blit(0, 132);
        cv_begin(240, 40, C_BLACK);          /* lit a moment after a turn */
        if (ui.page == PG_BAND) {            /* (K1 .. K4, left to right under the screen) */
            knob_cell(8, "TEMPO", s->bpm, "", lit && ui.turned == EN_K1);
            knob_cell(66, "FREEDOM", s->freedom, "", lit && ui.turned == EN_K2);
            knob_cell(126, "DRUMS", s->energy, "", lit && ui.turned == EN_K3);
            knob_cell(184, "SWING", s->swing, "%", lit && ui.turned == EN_K4);
        } else {
            knob_cell(8, "PIANO", s->vol_piano, "", lit && ui.turned == EN_K1);
            knob_cell(66, "BASS", s->vol_bass, "", lit && ui.turned == EN_K2);
            knob_cell(126, "DRUMS", s->vol_drums, "", lit && ui.turned == EN_K3);
            knob_cell(184, "ROOM", s->room, "", lit && ui.turned == EN_K4);
        }
        cv_blit(0, 172);
    }
    cv_begin(240, 24, C_BLACK);              /* each button's name and what it does on this page */
    {
        static const char *const BAND[8] = {"PLAY", "START", "ARP", "FEEL", "REC", "NEW TUNE", "GLO", "MIX"};
        static const char *const TEACH[8] = {"PLAY", "STOP", "EDIT", "LEARN IT", "GLO", "MIX", "", ""};   /* (no room for
                                                                                         * REC NEW TUNE too) */
        static const char *const MIX[8] = {"PRESET", "PIANO SOUND", "SAVE", "KEEP", "GLO", "BAND", "", ""};
        const char *bt[8] = {"PRESET", "BLUETOOTH", "SEL", "CONNECT", "GLO", "BAND", "", ""};
        const char *const *f = ui.page == PG_BT ? bt : ui.page == PG_MIX ? MIX : tr_teaching() ? TEACH : BAND;
        int32_t x = 10;
        int i;
        if (btdev_busy())
            bt[3] = "DISCONNECT";
        for (i = 0; i < 8 && f[i][0]; i += 2) {
            x = cv_text(x, 6, &FONT_T, f[i], C_GRAY);
            x = cv_text(x + 4, 6, &FONT_T, f[i + 1], C_DIM) + 12;
        }
    }
    cv_blit(0, 216);
}

void app_ui_init(void)
{
    touch_on(1);                             /* velocity from the keys */
    lcd_fill(0, 0, 240, 240, C_BLACK);
    ui.header = 1;                           /* (the first frame draws every strip) */
}

/* the screen in strips, each drawn again when what it shows changes: a beat redraws the header and the info strip
 * (80 rows), not the whole screen (a full screen is ~20 ms of DMA, and the networks share the main loop) */
enum { S_HEADER = 1, S_CHORD = 2, S_INFO = 4, S_REST = 8, S_ALL = 15 };

void app_frame(void)
{
    uint32_t now = plat_ms();
    int r = -1, q = -1, nr = -1, nq = -1, nin = 0, how, k, strips = 0;
    int32_t v[7];
    static const uint8_t STRIP[7] = {S_HEADER, S_CHORD, S_INFO, S_INFO, S_REST, S_HEADER | S_REST,
                                     S_HEADER | S_INFO | S_REST};
    input();
    ble_frame();
    speak();
    if (ui.learning && !tr_learning() && tr_learned())   /* the line learned: the bass walks it from now on */
        app_say("LEARNED", "YOUR LINE");
    ui.learning = (uint8_t)tr_learning();
    if (!tr_chord_now(&r, &q))
        r = q = -1;
    how = tr_chord_next(&nr, &nq, &nin);
    v[0] = tr_beat * 64 + (int32_t)(tr_missed > 63u ? 63u : tr_missed) + (int32_t)tr_state * 16;
    v[1] = r * 16 + q;
    v[2] = how * 65536 + nr * 4096 + nq * 256 + (nin > 255 ? 255 : nin);
    v[3] = tr_form_bars() * 256 + tr_bass_note() + 1;
    v[4] = tr_feel_now() + 4 * tr_state;
    v[5] = FM1B_BLE ? ble_on() | plat_ble_connected() << 1 | plat_ble_device() << 2 | (plat_ble_state() & 3) << 5 |
                          btdev_count() << 7 | btdev_pick() << 11 : 0;   /* Bluetooth: the header, its page */
    v[6] = tr_teaching() | tr_learning() << 1 | (tr_learned() > 0) << 2 | ((tr_taught() + 2) / 4) << 3;   /* EDIT's */
    for (k = 0; k < 7; k++)
        if (v[k] != ui.shown[k]) {
            ui.shown[k] = v[k];
            strips |= STRIP[k];
        }
    if (ui.redraw)                           /* a setting, a page, a message or a knob's light */
        strips |= S_HEADER | S_REST;
    if (ui.msg_until && now >= ui.msg_until) {   /* a message ends */
        ui.msg_until = 0;
        strips |= S_HEADER;
    }
    if (ui.turned_at && now - ui.turned_at >= 600u) {   /* a knob's light goes out */
        ui.turned_at = 0;
        strips |= S_REST;
    }
    if (ui.header)                           /* the first frame: everything */
        strips = S_ALL;
    ui.redraw = ui.header = 0;
    if (strips & S_HEADER)
        draw_header();
    if (strips & S_CHORD)
        draw_chord();
    if (strips & S_INFO)
        draw_info();
    if (strips & S_REST)
        draw_rest();
}

#ifdef PLAT_HOST
/* the host simulator's checks (tests/basic.txt) */
int app_expect(const char *what, int *got)
{
    int r = -1, q = -1, nr = -1, nq = -1, nin = 0;
    tr_chord_now(&r, &q);
    if (!strcmp(what, "state"))              /* 0 stopped, 1 count-in, 2 playing */
        *got = tr_state;
    else if (!strcmp(what, "beat"))
        *got = tr_beat;
    else if (!strcmp(what, "bpm"))
        *got = set.tr.bpm;
    else if (!strcmp(what, "feel"))
        *got = set.tr.feel;
    else if (!strcmp(what, "feel_now"))
        *got = tr_feel_now();
    else if (!strcmp(what, "freedom"))
        *got = set.tr.freedom;
    else if (!strcmp(what, "energy"))
        *got = set.tr.energy;
    else if (!strcmp(what, "root"))          /* the chord now: -1 none */
        *got = r;
    else if (!strcmp(what, "qual"))
        *got = q;
    else if (!strcmp(what, "next"))          /* 0 none, 1 from the form, 2 the network's guess */
        *got = tr_chord_next(&nr, &nq, &nin);
    else if (!strcmp(what, "form"))
        *got = tr_form_bars();
    else if (!strcmp(what, "bass"))
        *got = tr_bass_note();
    else if (!strcmp(what, "missed"))
        *got = (int)tr_missed;
    else if (!strcmp(what, "nnsteps"))
        *got = (int)tr_nn_steps();
    else if (!strcmp(what, "piano"))
        *got = set.tr.piano;
    else if (!strcmp(what, "vol_bass"))
        *got = set.tr.vol_bass;
    else if (!strcmp(what, "dirty"))
        *got = set_dirty;
    else if (!strcmp(what, "page"))
        *got = ui.page;
    else if (!strcmp(what, "octave"))
        *got = set.octave;
    else if (!strcmp(what, "ble_off"))       /* the setting (the platform's "ble": the radio's state) */
        *got = set.ble_off;
    else if (!strcmp(what, "ble_held")) {    /* notes held over Bluetooth (app_render) */
        int k;
        *got = 0;
        for (k = 0; k < 128; k++)
            *got += (int)(ble_held[k >> 5] >> (k & 31) & 1u);
    } else if (!strcmp(what, "pick"))        /* the device ALGO picked (lib/btdev) */
        *got = btdev_pick();
    else if (!strcmp(what, "let_go"))        /* SEL's let go, until the device is no longer busy */
        *got = let_go;
    else if (!strcmp(what, "teaching"))      /* EDIT: teaching the bass */
        *got = tr_teaching();
    else if (!strcmp(what, "learning"))      /* the main loop learning the line */
        *got = tr_learning();
    else if (!strcmp(what, "learned"))       /* lines learned into the bass (0: as trained) */
        *got = tr_learned();
    else if (!strcmp(what, "taught"))        /* beats the lesson keeps so far */
        *got = tr_taught();
    else if (!strcmp(what, "bass_unsaved"))
        *got = bass_unsaved();
    else if (!strcmp(what, "agree_before") || !strcmp(what, "agree_after")) {
        /* how often the bass's note was the one taught on the same beat of the form (%): over the chorus before the
         * last lesson, or from the beat it came into use until now (-1: nothing to compare) */
        int32_t first, last, lag = tr_form_bars() * 4;
        if (!tr_teach_span(&first, &last) || !lag)
            *got = -1;
        else if (what[6] == 'b')
            *got = tr_agree(first - lag, first, lag);
        else
            *got = tr_agree(tr_learned_at(), tr_beat, lag);
    } else
        return 0;
    return 1;
}

void app_host_reboot(void)               /* what a power cycle forgets: the screen's state, the unsaved, Bluetooth's */
{
    memset(&ui, 0, sizeof ui);
    set_dirty = 0;
    memset(ble_held, 0, sizeof ble_held);
    ble_pedal = ble_release = let_go = 0;
}
#endif
