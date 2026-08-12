/* test_npu_neuralart.c — host rig for fw/src/npu_neuralart.c.
 *
 * Runs the production driver against a FAKE graph whose shapes, scales and
 * zero points are taken from a real stedgeai report (profile n6-app-safe,
 * 3 Aug 2026): x[1,8,2,48] QLinear(0.004493613,-59), y[1,4,2,48]
 * QLinear(0.061152164,27), 12 state pairs [1,88,2,2*d] with their own
 * scales. LL_ATON is replaced by stubs in stub/ — the structure
 * LL_Buffer_InfoTypeDef and the signatures are copied from the real headers,
 * so a successful build proves the calls match the real API.
 *
 * What is caught here (and what no run on the board "by ear" will catch):
 *   - zeroing a voice with the value 0 instead of zero_point (gives a constant
 *     -zp*scale in every state channel);
 *   - a wrong voice slice stride in the [1,C,V,W] layout — clobbering ANOTHER
 *     voice is checked byte by byte;
 *   - loss or transposition of the balancing scales s_in/s_out;
 *   - divergence between the graph shape and n6_params_t (V, T, c_in, c_out).
 *
 * BUILD (from the repository root):
 *   gcc -std=c11 -Wall -Wextra -O1 \
 *       -Ifw/src -Ifw -Ifw/test/npu_iface/stub \
 *       fw/src/npu_neuralart.c fw/test/npu_iface/test_npu_neuralart.c \
 *       -o build/test_npu -lm && ./build/test_npu
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ll_aton_runtime.h"
#include "npu_iface.h"
#include "n6_npu_scales.h"

extern uint32_t n6_npu_prof_err, n6_npu_prof_epochs;

#define NB 13
#define V 2
#define T 48
#define C 88
static const uint32_t W[12] = {2,4,8,16,32,64,2,4,8,16,32,64};
static const float SSC[12] = {0.006209924f,0.006209924f,0.005856039f,0.007613949f,
                              0.007613949f,0.007951452f,0.011814119f,0.017094295f,
                              0.019347776f,0.027333392f,0.036022800f,0.063655034f};
static const int16_t SZP[12] = {-24,-24,-33,-55,-54,-58,-77,-92,-98,-108,-112,-119};

#define NPAR 5                    /* this many weight buffers are mixed in */
static LL_Buffer_InfoTypeDef IB[NB+NPAR+1], OB[NB+NPAR+1];
static uint32_t SH[NB+NPAR][4], SHO[NB+NPAR][4], MS[NB+NPAR][4], MSO[NB+NPAR][4];
static float SC_I[NB+NPAR], SC_O[NB+NPAR];
static int16_t ZP_I[NB+NPAR], ZP_O[NB+NPAR];
static unsigned char *MEM_I[NB+NPAR], *MEM_O[NB+NPAR];
static int epoch_n, fail;

static void mkbuf(LL_Buffer_InfoTypeDef *b, uint32_t sh[4], uint32_t ms[4],
                  unsigned char **mem, float *sc, int16_t *zp, const char *nm,
                  uint32_t c, uint32_t w, float s, int16_t z, uint8_t is_param) {
    /* AS IN network.c: shape uses a different axis order, mem_shape is
     * [1,C,V,W]. If the driver reads shape again, the test catches it. */
    sh[0]=1; sh[1]=V;  sh[2]=w; sh[3]=c;
    ms[0]=1; ms[1]=c;  ms[2]=V; ms[3]=w;
    uint32_t n = c*V*w;
    *mem = calloc(n,1); *sc = s; *zp = z;
    b->name=nm; b->addr_base=*mem; b->offset_start=0; b->offset_end=n;
    b->ndims=4; b->shape=sh; b->mem_ndims=4; b->mem_shape=ms;
    b->scale=sc; b->offset=zp; b->per_channel=0; b->is_param=is_param;
}
static void build(void) {
    mkbuf(&IB[0],SH[0],MS[0],&MEM_I[0],&SC_I[0],&ZP_I[0],"x",8,T,0.004493613f,-59,0);
    mkbuf(&OB[0],SHO[0],MSO[0],&MEM_O[0],&SC_O[0],&ZP_O[0],"y",4,T,0.061152164f,27,0);
    for (int k=0;k<12;k++){
        char *ni=malloc(24), *no=malloc(24);
        sprintf(ni,"si%d",k); sprintf(no,"so%d",k);
        mkbuf(&IB[k+1],SH[k+1],MS[k+1],&MEM_I[k+1],&SC_I[k+1],&ZP_I[k+1],ni,C,W[k],SSC[k],SZP[k],0);
        mkbuf(&OB[k+1],SHO[k+1],MSO[k+1],&MEM_O[k+1],&SC_O[k+1],&ZP_O[k+1],no,C,W[k],SSC[k],SZP[k],0);
    }
    /* WEIGHTS AND BIASES: in network.c they sit in the same list with
     * is_param=1. They are exactly what the driver tripped over on the board
     * on 3 Aug (prof_err=1). */
    for (int k=0;k<NPAR;k++){
        char *pi=malloc(32), *po=malloc(32);
        sprintf(pi,"Conv2D_%d_weights",k); sprintf(po,"Conv2D_%d_bias",k);
        mkbuf(&IB[NB+k],SH[NB+k],MS[NB+k],&MEM_I[NB+k],&SC_I[NB+k],&ZP_I[NB+k],pi,4,4,1.0f,0,1);
        mkbuf(&OB[NB+k],SHO[NB+k],MSO[NB+k],&MEM_O[NB+k],&SC_O[NB+k],&ZP_O[NB+k],po,4,4,1.0f,0,1);
    }
    IB[NB+NPAR].name=NULL; OB[NB+NPAR].name=NULL;
}
void LL_ATON_RT_RuntimeInit(void){}
void LL_ATON_RT_Init_Network(NN_Instance_TypeDef *i){(void)i;}
void LL_ATON_RT_Reset_Network(NN_Instance_TypeDef *i){(void)i;epoch_n=0;}
LL_ATON_RT_RetValues_t LL_ATON_RT_RunEpochBlock(NN_Instance_TypeDef *i){
    (void)i;
    if (++epoch_n < 3) return (epoch_n==1)?LL_ATON_RT_NO_WFE:LL_ATON_RT_WFE;
    /* "compute": y[b][v][t] = x[b][v][t] (the same int8 codes), states +1 */
    memcpy(MEM_O[0], MEM_I[0], 4*V*T);
    for (int k=0;k<12;k++){ uint32_t n=C*V*W[k];
        for (uint32_t j=0;j<n;j++) MEM_O[k+1][j]=(unsigned char)(MEM_I[k+1][j]+1); }
    return LL_ATON_RT_DONE;
}
const LL_Buffer_InfoTypeDef *LL_ATON_Input_Buffers_Info(const NN_Instance_TypeDef *i){(void)i;return IB;}
const LL_Buffer_InfoTypeDef *LL_ATON_Output_Buffers_Info(const NN_Instance_TypeDef *i){(void)i;return OB;}

#define CHECK(c,msg,...) do{ if(!(c)){printf("  FAIL: " msg "\n",##__VA_ARGS__); fail++;} }while(0)

int main(void){
    setvbuf(stdout,NULL,_IOLBF,0);
    build();
    n6_params_t prm = { V, N6_HOP48, 2.0f, 0.06f };
    n6_npu_t *n = n6_npu_create(&prm);
    printf("1. create: prof_err=%u (expect 0; the list holds %d inputs + %d weights)\n",
           n6_npu_prof_err, NB, NPAR);
    CHECK(n6_npu_prof_err==0,"initialisation failed the shape contract");
    if (n6_npu_prof_err) {          /* cannot go on: res is not allocated */
        if (n6_npu_prof_err >= 1000)
            printf("\n   decoding: 1000+in*100+out => in=%u out=%u (expect 13/13)\n",
                   (n6_npu_prof_err-1000)/100, (n6_npu_prof_err-1000)%100);
        else
            printf("\n   code %u — see the decoding in the create() header (2/3 shape,"
                   " 4 V or T, 5 size, 6 quant, 10+/30+/50+ state pair)\n",
                   n6_npu_prof_err);
        printf("\nTHERE ARE FAILURES (the remaining checks were not run)\n");
        return 1;
    }

    printf("2. states start at zero_point, not at a zero byte\n");
    for(int k=0;k<12;k++){
        CHECK(((signed char*)MEM_I[k+1])[0]==SZP[k],
              "  state %d: %d instead of %d",k,((signed char*)MEM_I[k+1])[0],SZP[k]);
    }

    printf("3. input quantisation: divide by s_in[b], then by scale, plus zp\n");
    float *xc = calloc(8*V*T,sizeof(float));
    for(int c=0;c<8;c++) for(int i=0;i<V*T;i++) xc[c*V*T+i] = 0.02f*((c%3)-1) + 0.001f*i;
    n6_npu_submit(n, xc);
    int bad=0; float worst=0;
    for(int c=0;c<8;c++) for(int i=0;i<V*T;i++){
        float g = (c<4)? (1.0f/(0.004493613f*n6_npu_s_in[c])) : (1.0f/0.004493613f);
        int q = (int)lrintf(xc[c*V*T+i]*g) + (-59);
        if(q<-128){q=-128;}
        if(q>127){q=127;}
        int got = ((signed char*)MEM_I[0])[c*V*T+i];
        if(got!=q){ bad++; if(fabsf((float)(got-q))>worst) worst=fabsf((float)(got-q)); }
    }
    CHECK(bad==0,"  quantisation mismatches: %d (max %g)",bad,worst);

    printf("4. pumping: poll until DONE\n");
    int polls=0; while(!n6_npu_poll(n) && polls<100) polls++;
    printf("   epoch blocks per hop: %u, polls: %d\n", n6_npu_prof_epochs, polls+1);
    CHECK(polls+1==3,"  expected 3 calls, got %d",polls+1);

    printf("5. output dequantisation: (code-zp)*scale*s_out[b]\n");
    const float *res = n6_npu_residual(n);
    bad=0; double worst_rel=0, worst_abs=0, peak=0;
    for(int b=0;b<4;b++) for(int i=0;i<V*T;i++){
        double want = (double)(((signed char*)MEM_O[0])[b*V*T+i]-27)*0.061152164*(double)n6_npu_s_out[b];
        double got  = res[b*V*T+i];
        double a = fabs(got-want), r = a/(fabs(want)+1e-30);
        if(fabs(want)>peak) peak=fabs(want);
        if(a>worst_abs) worst_abs=a;
        if(fabs(want)>1e-6 && r>worst_rel) worst_rel=r;
        if(fabs(want)>1e-6 && r > 1e-6) bad++;
    }
    printf("   peak |residual| = %.4g; max abs %.3g, max rel %.3g\n", peak, worst_abs, worst_rel);
    CHECK(bad==0,"  dequantisation mismatches beyond float rounding: %d",bad);

    printf("6. zero_voice: we zero ONLY the slice of voice 1, with zero_point\n");
    for(int k=0;k<12;k++) memset(MEM_O[k+1],0x5A,C*V*W[k]);
    n6_npu_zero_voice(n,1);
    int touched_wrong=0, missed=0;
    for(int k=0;k<12;k++){
        signed char *p=(signed char*)MEM_O[k+1];
        for(uint32_t c=0;c<C;c++) for(uint32_t v=0;v<V;v++) for(uint32_t w=0;w<W[k];w++){
            signed char got = p[(c*V+v)*W[k]+w];
            if(v==1){ if(got!=SZP[k]) missed++; }
            else    { if(got!=0x5A)   touched_wrong++; }
        }
    }
    CHECK(missed==0,"  bytes not zeroed in voice 1: %d",missed);
    CHECK(touched_wrong==0,"  ANOTHER VOICE WAS TOUCHED: %d bytes",touched_wrong);

    printf("7. swap_states: the input ring becomes the output ring\n");
    n6_npu_swap_states(n);
    bad=0; for(int k=0;k<12;k++) if(memcmp(MEM_I[k+1],MEM_O[k+1],C*V*W[k])) bad++;
    CHECK(bad==0,"  pairs not copied: %d",bad);

    printf("8. guard against shape divergence: we feed 3 voices instead of 2\n");
    n6_npu_prof_err=0;
    n6_params_t bad_prm = { 3, N6_HOP48, 2.0f, 0.06f };
    n6_npu_create(&bad_prm);
    printf("   prof_err=%u (expect 4 — the graph and the runtime diverged)\n", n6_npu_prof_err);
    CHECK(n6_npu_prof_err==4,"  the V mismatch was not caught");

    printf("\n%s\n", fail? "THERE ARE FAILURES" : "ALL CHECKS PASSED");
    return fail!=0;
}
