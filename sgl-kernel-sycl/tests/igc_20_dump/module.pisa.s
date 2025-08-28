	.file	"module.pisa"
	.section	.text._ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE0_clES5_EUlNS3_7nd_itemILi3EEEE_,"ax",@progbits
	.type	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE0_clES5_EUlNS3_7nd_itemILi3EEEE_,@function // -- Begin function _ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE0_clES5_EUlNS3_7nd_itemILi3EEEE_
_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE0_clES5_EUlNS3_7nd_itemILi3EEEE_: // @_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE0_clES5_EUlNS3_7nd_itemILi3EEEE_
// %bb.0:
	gctrl = sbfia.(2,6).b32 gctrl, 0x3 	{NoDep, 128G}
	gctrl = sbfia.(1,10).b32 gctrl, 0x1 	{NoDep, 128G}
	gctrl = sbfia.(1,30).b32 gctrl, 0x1 	{NoDep, 128G}
	r2 = shl.b32 r1, 0x2 	{NoDep, 128A}
	r3 = mov.b32 0x0 	{NoDep, 64I}
	r4 = shr.b32 r0, 0x3 	{NoDep, 128A}
	r5 = mov.b32 r3 	{int@2, 64I}
	(s8,s9) = smullh.u64 s5, 0x18 	{NoDep, 128A}
	send.slm.fence.tg 	{$0, NoDep, 128C}
	sync.none 	{$0.dst, 64E}
	send.ugm.fence.tg 	{$1, NoDep, 128C}
	sync.none 	{$1.dst, 64E}
	(r2,r3) = add3.s64 (s8,s9), (r4,r5), (r2,r3) 	{int@1, scl@1, 128A}
	send.gtwy.bar s0 	{$2, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.slm.fence.tg 	{$3, NoDep, 128C}
	sync.none 	{$3.dst, 64E}
	send.ugm.fence.tg 	{$4, NoDep, 128C}
	sync.none 	{$4.dst, 64E}
	send.gtwy.bar s0 	{$5, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	p0 = cmp.b32::ge (r2,r3).s64, (s28,s29) 	{int@1, 64B}
	send.slm.fence.tg 	{$6, NoDep, 128C}
	sync.none 	{$6.dst, 64E}
	send.ugm.fence.tg 	{$7, NoDep, 128C}
	sync.none 	{$7.dst, 64E}
	send.gtwy.bar s0 	{$8, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	sync.none 	{all, 64E}
	sync.none 	{$8.src, 64E}
	(p0) goto LBB0_5, LBB0_5 	{NoDep, 128B}
// %bb.1:
	(s2,s3) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 168] 	{$9, NoDep, 128C}
	p0 = cmp.b32::le (s2,s3).s64, 0x0 	{$9.dst, 128S}
	s4 = smov.b32 0xffffffff 	{NoDep, 128R}
	r0, p1 = bfn3.(s0&s1).b32::ne r0, 0x7, null 	{NoDep, 128E}
	r0 = mov.b32 0xffffffff 	{NoDep, 128R}
	s4 = ssel.b32 s4, 0x0, p0 	{scl@1, 128J}
	r0 = sel.b32 r0, 0x0, p1 	{int@1, 128J}
	sync.none 	{scl@1, 64E}
	r0 = bfn2.(s0|s1).b32 s4, r0 	{int@1, 64D}
	r0, p0 = bfn3.(s0&s1).b32::ne r0, 0x1, null 	{int@1, 128E}
	sync.none 	{all, 64E}
	(p0) goto LBB0_5, LBB0_5 	{NoDep, 128B}
// %bb.2:
	(s8,s9) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 160] 	{$10, NoDep, 128C}
	r0 = mov.b32 s8 	{$10.dst, 64I}
	(r4,r5) = mullh.u64 r2, r0 	{int@1, 128A}
	r1 = mov.b32 s9 	{NoDep, 64I}
	r1 = mad.u32 r2, r1, r5 	{int@1, 64C}
	r5 = mad.u32 r3, r0, r1 	{int@1, 64C}
	(r0,r1) = add3.s64 (s8,s9), -(s2,s3), (r4,r5) 	{int@1, 128A}
	(r2,r3) = shl.b64 (r0,r1), 0x2 	{int@1, 128A}
	(r4,r5) = add.u64 (s26,s27), (r2,r3) 	{int@1, 64A}
	(r2,r3) = add.u64 (s24,s25), (r2,r3) 	{NoDep, 64A}
	p0 = cmp.b32::le (s2,s3).u64, 0x1 	{NoDep, 128S}
	r8 = mov.b32 0x100 	{NoDep, 64I}
	s9 = smov.b32 0x0 	{NoDep, 64I}
	r9 = mov.b32 0x0 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r4,r5) * 1], r8 	{$11, int@2, 128C}
	send.ugm.st.d32.a64 [(r2,r3) * 1], r9 	{$12, int@1, 128C}
	sync.none 	{all, 64E}
	(p0) goto LBB0_5, LBB0_5 	{NoDep, 128B}
// %bb.3:
	s4 = smov.b32 0x1 	{NoDep, 64I}
	r2 = mov.b32 s9 	{$12.src, 64I}
	r3 = mov.b32 s9 	{NoDep, 64I}
	r4 = mov.b32 s4 	{scl@1, $11.src, 128R}
	sync.none 	{all, 64E}
LBB0_4:                                 // =>This Inner Loop Header: Depth=1
	(r0,r1) = add.s64 (r0,r1), 0x1 	{NoDep, 64A}
	(r8,r9) = shl.b64 (r0,r1), 0x2 	{int@1, $14.src, 128A}
	s4 = redfirst.b32 r4, 0xffffffff 	{NoDep, 128F}
	s10 = redfirst.b32 r3, 0xffffffff 	{NoDep, 128F}
	(r10,r11) = add.u64 (s26,s27), (r8,r9) 	{int@1, $13.src, 128A}
	s8 = sadd.s32 s4, 0x1 	{shfl@2, 64A}
	(r8,r9) = add.u64 (s24,s25), (r8,r9) 	{NoDep, 64A}
	s4 = sadd.s32 s10, 0x101 	{shfl@1, 64A}
	s10 = sadd.s32 s10, 0x1 	{NoDep, 64A}
	p0 = cmp.b32::gt (s2,s3).u64, (s8,s9) 	{scl@3, 64B}
	r5 = mov.b32 s4 	{scl@2, 64I}
	r3 = mov.b32 s10 	{scl@1, 64I}
	r4 = mov.b32 s8 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r10,r11) * 1], r5 	{$13, int@3, 128C}
	send.ugm.st.d32.a64 [(r8,r9) * 1], r2 	{$14, NoDep, 128C}
	sync.none 	{all, 64E}
	(p0) goto::b LBB0_5, LBB0_4 	{NoDep, 128B}
LBB0_5:
	join LBB0_5 	{NoDep, 128B}
	send.slm.fence.tg 	{$15, NoDep, 128C}
	sync.none 	{$15.dst, 64E}
	send.ugm.fence.tg 	{$16, NoDep, 128C}
	sync.none 	{$16.dst, 64E}
	send.gtwy.bar s0 	{$17, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.gtwy.eot s0 	{$18, NoDep, 128C}
Lfunc_end0:                             // -- Function level Xe statistics
                                        // VRT: NumGRFs=16, NumSRFs=32
                                        // NumInsts: 91
                                        // NumCompactInsts: 44
                                        // NumSyncInsts: 19
                                        // NumScalarInsts: 11
                                        // NumSRF2GRFCopies: 8
                                        // NumGRF2SRFCopies: 2
                                        // NumGRFSpillInsts: 0
                                        // NumGRFFillInsts: 0
                                        // NumSatInsts: 0
                                        // MaxGRFUsed: 12
                                        // MaxSRFUsed: 30
                                        // NumExtendedSRFLiveRange: 4
                                        // NumCycles: 268
                                        // GRFBypassEntriesPerThread: 0
	.size	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE0_clES5_EUlNS3_7nd_itemILi3EEEE_, Lfunc_end0-_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE0_clES5_EUlNS3_7nd_itemILi3EEEE_
                                        // -- End function
	.section	.text._ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE3_clES5_EUlNS3_7nd_itemILi3EEEE_,"ax",@progbits
	.type	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE3_clES5_EUlNS3_7nd_itemILi3EEEE_,@function // -- Begin function _ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE3_clES5_EUlNS3_7nd_itemILi3EEEE_
_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE3_clES5_EUlNS3_7nd_itemILi3EEEE_: // @_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE3_clES5_EUlNS3_7nd_itemILi3EEEE_
// %bb.0:
	gctrl = sbfia.(2,6).b32 gctrl, 0x3 	{NoDep, 128G}
	gctrl = sbfia.(1,10).b32 gctrl, 0x1 	{NoDep, 128G}
	gctrl = sbfia.(1,30).b32 gctrl, 0x1 	{NoDep, 128G}
	r2 = shl.b32 r1, 0x1 	{NoDep, 128A}
	r3 = mov.b32 0x0 	{NoDep, 64I}
	r4 = shr.b32 r0, 0x4 	{NoDep, 128A}
	r5 = mov.b32 r3 	{int@2, 64I}
	(s8,s9) = smullh.u64 s5, 0xc 	{NoDep, 128A}
	send.slm.fence.tg 	{$0, NoDep, 128C}
	sync.none 	{$0.dst, 64E}
	send.ugm.fence.tg 	{$1, NoDep, 128C}
	sync.none 	{$1.dst, 64E}
	(r2,r3) = add3.s64 (s8,s9), (r4,r5), (r2,r3) 	{int@1, scl@1, 128A}
	send.gtwy.bar s0 	{$2, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.slm.fence.tg 	{$3, NoDep, 128C}
	sync.none 	{$3.dst, 64E}
	send.ugm.fence.tg 	{$4, NoDep, 128C}
	sync.none 	{$4.dst, 64E}
	send.gtwy.bar s0 	{$5, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	p0 = cmp.b32::ge (r2,r3).s64, (s28,s29) 	{int@1, 64B}
	send.slm.fence.tg 	{$6, NoDep, 128C}
	sync.none 	{$6.dst, 64E}
	send.ugm.fence.tg 	{$7, NoDep, 128C}
	sync.none 	{$7.dst, 64E}
	send.gtwy.bar s0 	{$8, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	sync.none 	{all, 64E}
	sync.none 	{$8.src, 64E}
	(p0) goto LBB1_5, LBB1_5 	{NoDep, 128B}
// %bb.1:
	(s2,s3) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 168] 	{$9, NoDep, 128C}
	p0 = cmp.b32::le (s2,s3).s64, 0x0 	{$9.dst, 128S}
	s4 = smov.b32 0xffffffff 	{NoDep, 128R}
	r0, p1 = bfn3.(s0&s1).b32::ne r0, 0xf, null 	{NoDep, 128E}
	r0 = mov.b32 0xffffffff 	{NoDep, 128R}
	s4 = ssel.b32 s4, 0x0, p0 	{scl@1, 128J}
	r0 = sel.b32 r0, 0x0, p1 	{int@1, 128J}
	sync.none 	{scl@1, 64E}
	r0 = bfn2.(s0|s1).b32 s4, r0 	{int@1, 64D}
	r0, p0 = bfn3.(s0&s1).b32::ne r0, 0x1, null 	{int@1, 128E}
	sync.none 	{all, 64E}
	(p0) goto LBB1_5, LBB1_5 	{NoDep, 128B}
// %bb.2:
	(s8,s9) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 160] 	{$10, NoDep, 128C}
	r0 = mov.b32 s8 	{$10.dst, 64I}
	(r4,r5) = mullh.u64 r2, r0 	{int@1, 128A}
	r1 = mov.b32 s9 	{NoDep, 64I}
	r1 = mad.u32 r2, r1, r5 	{int@1, 64C}
	r5 = mad.u32 r3, r0, r1 	{int@1, 64C}
	(r0,r1) = add3.s64 (s8,s9), -(s2,s3), (r4,r5) 	{int@1, 128A}
	(r2,r3) = shl.b64 (r0,r1), 0x2 	{int@1, 128A}
	(r4,r5) = add.u64 (s26,s27), (r2,r3) 	{int@1, 64A}
	(r2,r3) = add.u64 (s24,s25), (r2,r3) 	{NoDep, 64A}
	p0 = cmp.b32::le (s2,s3).u64, 0x1 	{NoDep, 128S}
	r8 = mov.b32 0x100 	{NoDep, 64I}
	s9 = smov.b32 0x0 	{NoDep, 64I}
	r9 = mov.b32 0x0 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r4,r5) * 1], r8 	{$11, int@2, 128C}
	send.ugm.st.d32.a64 [(r2,r3) * 1], r9 	{$12, int@1, 128C}
	sync.none 	{all, 64E}
	(p0) goto LBB1_5, LBB1_5 	{NoDep, 128B}
// %bb.3:
	s4 = smov.b32 0x1 	{NoDep, 64I}
	r2 = mov.b32 s9 	{$12.src, 64I}
	r3 = mov.b32 s9 	{NoDep, 64I}
	r4 = mov.b32 s4 	{scl@1, $11.src, 128R}
	sync.none 	{all, 64E}
LBB1_4:                                 // =>This Inner Loop Header: Depth=1
	(r0,r1) = add.s64 (r0,r1), 0x1 	{NoDep, 64A}
	(r8,r9) = shl.b64 (r0,r1), 0x2 	{int@1, $14.src, 128A}
	s4 = redfirst.b32 r4, 0xffffffff 	{NoDep, 128F}
	s10 = redfirst.b32 r3, 0xffffffff 	{NoDep, 128F}
	(r10,r11) = add.u64 (s26,s27), (r8,r9) 	{int@1, $13.src, 128A}
	s8 = sadd.s32 s4, 0x1 	{shfl@2, 64A}
	(r8,r9) = add.u64 (s24,s25), (r8,r9) 	{NoDep, 64A}
	s4 = sadd.s32 s10, 0x101 	{shfl@1, 64A}
	s10 = sadd.s32 s10, 0x1 	{NoDep, 64A}
	p0 = cmp.b32::gt (s2,s3).u64, (s8,s9) 	{scl@3, 64B}
	r5 = mov.b32 s4 	{scl@2, 64I}
	r3 = mov.b32 s10 	{scl@1, 64I}
	r4 = mov.b32 s8 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r10,r11) * 1], r5 	{$13, int@3, 128C}
	send.ugm.st.d32.a64 [(r8,r9) * 1], r2 	{$14, NoDep, 128C}
	sync.none 	{all, 64E}
	(p0) goto::b LBB1_5, LBB1_4 	{NoDep, 128B}
LBB1_5:
	join LBB1_5 	{NoDep, 128B}
	send.slm.fence.tg 	{$15, NoDep, 128C}
	sync.none 	{$15.dst, 64E}
	send.ugm.fence.tg 	{$16, NoDep, 128C}
	sync.none 	{$16.dst, 64E}
	send.gtwy.bar s0 	{$17, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.gtwy.eot s0 	{$18, NoDep, 128C}
Lfunc_end1:                             // -- Function level Xe statistics
                                        // VRT: NumGRFs=16, NumSRFs=32
                                        // NumInsts: 91
                                        // NumCompactInsts: 44
                                        // NumSyncInsts: 19
                                        // NumScalarInsts: 11
                                        // NumSRF2GRFCopies: 8
                                        // NumGRF2SRFCopies: 2
                                        // NumGRFSpillInsts: 0
                                        // NumGRFFillInsts: 0
                                        // NumSatInsts: 0
                                        // MaxGRFUsed: 12
                                        // MaxSRFUsed: 30
                                        // NumExtendedSRFLiveRange: 4
                                        // NumCycles: 268
                                        // GRFBypassEntriesPerThread: 0
	.size	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE3_clES5_EUlNS3_7nd_itemILi3EEEE_, Lfunc_end1-_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE3_clES5_EUlNS3_7nd_itemILi3EEEE_
                                        // -- End function
	.section	.text._ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE6_clES5_EUlNS3_7nd_itemILi3EEEE_,"ax",@progbits
	.type	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE6_clES5_EUlNS3_7nd_itemILi3EEEE_,@function // -- Begin function _ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE6_clES5_EUlNS3_7nd_itemILi3EEEE_
_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE6_clES5_EUlNS3_7nd_itemILi3EEEE_: // @_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE6_clES5_EUlNS3_7nd_itemILi3EEEE_
// %bb.0:
	gctrl = sbfia.(2,6).b32 gctrl, 0x3 	{NoDep, 128G}
	gctrl = sbfia.(1,10).b32 gctrl, 0x1 	{NoDep, 128G}
	gctrl = sbfia.(1,30).b32 gctrl, 0x1 	{NoDep, 128G}
	r2 = shl.b32 r1, 0x3 	{NoDep, 128A}
	r3 = mov.b32 0x0 	{NoDep, 64I}
	r4 = shr.b32 r0, 0x2 	{NoDep, 128A}
	r5 = mov.b32 r3 	{int@2, 64I}
	(s8,s9) = smullh.u64 s5, 0x30 	{NoDep, 128A}
	send.slm.fence.tg 	{$0, NoDep, 128C}
	sync.none 	{$0.dst, 64E}
	send.ugm.fence.tg 	{$1, NoDep, 128C}
	sync.none 	{$1.dst, 64E}
	(r2,r3) = add3.s64 (s8,s9), (r4,r5), (r2,r3) 	{int@1, scl@1, 128A}
	send.gtwy.bar s0 	{$2, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.slm.fence.tg 	{$3, NoDep, 128C}
	sync.none 	{$3.dst, 64E}
	send.ugm.fence.tg 	{$4, NoDep, 128C}
	sync.none 	{$4.dst, 64E}
	send.gtwy.bar s0 	{$5, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	p0 = cmp.b32::ge (r2,r3).s64, (s28,s29) 	{int@1, 64B}
	send.slm.fence.tg 	{$6, NoDep, 128C}
	sync.none 	{$6.dst, 64E}
	send.ugm.fence.tg 	{$7, NoDep, 128C}
	sync.none 	{$7.dst, 64E}
	send.gtwy.bar s0 	{$8, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	sync.none 	{all, 64E}
	sync.none 	{$8.src, 64E}
	(p0) goto LBB2_5, LBB2_5 	{NoDep, 128B}
// %bb.1:
	(s2,s3) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 168] 	{$9, NoDep, 128C}
	p0 = cmp.b32::le (s2,s3).s64, 0x0 	{$9.dst, 128S}
	s4 = smov.b32 0xffffffff 	{NoDep, 128R}
	r0, p1 = bfn3.(s0&s1).b32::ne r0, 0x3, null 	{NoDep, 128E}
	r0 = mov.b32 0xffffffff 	{NoDep, 128R}
	s4 = ssel.b32 s4, 0x0, p0 	{scl@1, 128J}
	r0 = sel.b32 r0, 0x0, p1 	{int@1, 128J}
	sync.none 	{scl@1, 64E}
	r0 = bfn2.(s0|s1).b32 s4, r0 	{int@1, 64D}
	r0, p0 = bfn3.(s0&s1).b32::ne r0, 0x1, null 	{int@1, 128E}
	sync.none 	{all, 64E}
	(p0) goto LBB2_5, LBB2_5 	{NoDep, 128B}
// %bb.2:
	(s8,s9) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 160] 	{$10, NoDep, 128C}
	r0 = mov.b32 s8 	{$10.dst, 64I}
	(r4,r5) = mullh.u64 r2, r0 	{int@1, 128A}
	r1 = mov.b32 s9 	{NoDep, 64I}
	r1 = mad.u32 r2, r1, r5 	{int@1, 64C}
	r5 = mad.u32 r3, r0, r1 	{int@1, 64C}
	(r0,r1) = add3.s64 (s8,s9), -(s2,s3), (r4,r5) 	{int@1, 128A}
	(r2,r3) = shl.b64 (r0,r1), 0x2 	{int@1, 128A}
	(r4,r5) = add.u64 (s26,s27), (r2,r3) 	{int@1, 64A}
	(r2,r3) = add.u64 (s24,s25), (r2,r3) 	{NoDep, 64A}
	p0 = cmp.b32::le (s2,s3).u64, 0x1 	{NoDep, 128S}
	r8 = mov.b32 0x80 	{NoDep, 64I}
	s9 = smov.b32 0x0 	{NoDep, 64I}
	r9 = mov.b32 0x0 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r4,r5) * 1], r8 	{$11, int@2, 128C}
	send.ugm.st.d32.a64 [(r2,r3) * 1], r9 	{$12, int@1, 128C}
	sync.none 	{all, 64E}
	(p0) goto LBB2_5, LBB2_5 	{NoDep, 128B}
// %bb.3:
	s4 = smov.b32 0x1 	{NoDep, 64I}
	r2 = mov.b32 s9 	{$12.src, 64I}
	r3 = mov.b32 s9 	{NoDep, 64I}
	r4 = mov.b32 s4 	{scl@1, $11.src, 128R}
	sync.none 	{all, 64E}
LBB2_4:                                 // =>This Inner Loop Header: Depth=1
	(r0,r1) = add.s64 (r0,r1), 0x1 	{NoDep, 64A}
	(r8,r9) = shl.b64 (r0,r1), 0x2 	{int@1, $14.src, 128A}
	s4 = redfirst.b32 r4, 0xffffffff 	{NoDep, 128F}
	s10 = redfirst.b32 r3, 0xffffffff 	{NoDep, 128F}
	(r10,r11) = add.u64 (s26,s27), (r8,r9) 	{int@1, $13.src, 128A}
	s8 = sadd.s32 s4, 0x1 	{shfl@2, 64A}
	(r8,r9) = add.u64 (s24,s25), (r8,r9) 	{NoDep, 64A}
	s4 = sadd.s32 s10, 0x81 	{shfl@1, 64A}
	s10 = sadd.s32 s10, 0x1 	{NoDep, 64A}
	p0 = cmp.b32::gt (s2,s3).u64, (s8,s9) 	{scl@3, 64B}
	r5 = mov.b32 s4 	{scl@2, 64I}
	r3 = mov.b32 s10 	{scl@1, 64I}
	r4 = mov.b32 s8 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r10,r11) * 1], r5 	{$13, int@3, 128C}
	send.ugm.st.d32.a64 [(r8,r9) * 1], r2 	{$14, NoDep, 128C}
	sync.none 	{all, 64E}
	(p0) goto::b LBB2_5, LBB2_4 	{NoDep, 128B}
LBB2_5:
	join LBB2_5 	{NoDep, 128B}
	send.slm.fence.tg 	{$15, NoDep, 128C}
	sync.none 	{$15.dst, 64E}
	send.ugm.fence.tg 	{$16, NoDep, 128C}
	sync.none 	{$16.dst, 64E}
	send.gtwy.bar s0 	{$17, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.gtwy.eot s0 	{$18, NoDep, 128C}
Lfunc_end2:                             // -- Function level Xe statistics
                                        // VRT: NumGRFs=16, NumSRFs=32
                                        // NumInsts: 91
                                        // NumCompactInsts: 44
                                        // NumSyncInsts: 19
                                        // NumScalarInsts: 11
                                        // NumSRF2GRFCopies: 8
                                        // NumGRF2SRFCopies: 2
                                        // NumGRFSpillInsts: 0
                                        // NumGRFFillInsts: 0
                                        // NumSatInsts: 0
                                        // MaxGRFUsed: 12
                                        // MaxSRFUsed: 30
                                        // NumExtendedSRFLiveRange: 4
                                        // NumCycles: 268
                                        // GRFBypassEntriesPerThread: 0
	.size	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE6_clES5_EUlNS3_7nd_itemILi3EEEE_, Lfunc_end2-_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE6_clES5_EUlNS3_7nd_itemILi3EEEE_
                                        // -- End function
	.section	.text._ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE9_clES5_EUlNS3_7nd_itemILi3EEEE_,"ax",@progbits
	.type	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE9_clES5_EUlNS3_7nd_itemILi3EEEE_,@function // -- Begin function _ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE9_clES5_EUlNS3_7nd_itemILi3EEEE_
_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE9_clES5_EUlNS3_7nd_itemILi3EEEE_: // @_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE9_clES5_EUlNS3_7nd_itemILi3EEEE_
// %bb.0:
	gctrl = sbfia.(2,6).b32 gctrl, 0x3 	{NoDep, 128G}
	gctrl = sbfia.(1,10).b32 gctrl, 0x1 	{NoDep, 128G}
	gctrl = sbfia.(1,30).b32 gctrl, 0x1 	{NoDep, 128G}
	r2 = shl.b32 r1, 0x2 	{NoDep, 128A}
	r3 = mov.b32 0x0 	{NoDep, 64I}
	r4 = shr.b32 r0, 0x3 	{NoDep, 128A}
	r5 = mov.b32 r3 	{int@2, 64I}
	(s8,s9) = smullh.u64 s5, 0x18 	{NoDep, 128A}
	send.slm.fence.tg 	{$0, NoDep, 128C}
	sync.none 	{$0.dst, 64E}
	send.ugm.fence.tg 	{$1, NoDep, 128C}
	sync.none 	{$1.dst, 64E}
	(r2,r3) = add3.s64 (s8,s9), (r4,r5), (r2,r3) 	{int@1, scl@1, 128A}
	send.gtwy.bar s0 	{$2, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.slm.fence.tg 	{$3, NoDep, 128C}
	sync.none 	{$3.dst, 64E}
	send.ugm.fence.tg 	{$4, NoDep, 128C}
	sync.none 	{$4.dst, 64E}
	send.gtwy.bar s0 	{$5, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	p0 = cmp.b32::ge (r2,r3).s64, (s28,s29) 	{int@1, 64B}
	send.slm.fence.tg 	{$6, NoDep, 128C}
	sync.none 	{$6.dst, 64E}
	send.ugm.fence.tg 	{$7, NoDep, 128C}
	sync.none 	{$7.dst, 64E}
	send.gtwy.bar s0 	{$8, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	sync.none 	{all, 64E}
	sync.none 	{$8.src, 64E}
	(p0) goto LBB3_5, LBB3_5 	{NoDep, 128B}
// %bb.1:
	(s2,s3) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 168] 	{$9, NoDep, 128C}
	p0 = cmp.b32::le (s2,s3).s64, 0x0 	{$9.dst, 128S}
	s4 = smov.b32 0xffffffff 	{NoDep, 128R}
	r0, p1 = bfn3.(s0&s1).b32::ne r0, 0x7, null 	{NoDep, 128E}
	r0 = mov.b32 0xffffffff 	{NoDep, 128R}
	s4 = ssel.b32 s4, 0x0, p0 	{scl@1, 128J}
	r0 = sel.b32 r0, 0x0, p1 	{int@1, 128J}
	sync.none 	{scl@1, 64E}
	r0 = bfn2.(s0|s1).b32 s4, r0 	{int@1, 64D}
	r0, p0 = bfn3.(s0&s1).b32::ne r0, 0x1, null 	{int@1, 128E}
	sync.none 	{all, 64E}
	(p0) goto LBB3_5, LBB3_5 	{NoDep, 128B}
// %bb.2:
	(s8,s9) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 160] 	{$10, NoDep, 128C}
	r0 = mov.b32 s8 	{$10.dst, 64I}
	(r4,r5) = mullh.u64 r2, r0 	{int@1, 128A}
	r1 = mov.b32 s9 	{NoDep, 64I}
	r1 = mad.u32 r2, r1, r5 	{int@1, 64C}
	r5 = mad.u32 r3, r0, r1 	{int@1, 64C}
	(r0,r1) = add3.s64 (s8,s9), -(s2,s3), (r4,r5) 	{int@1, 128A}
	(r2,r3) = shl.b64 (r0,r1), 0x2 	{int@1, 128A}
	(r4,r5) = add.u64 (s26,s27), (r2,r3) 	{int@1, 64A}
	(r2,r3) = add.u64 (s24,s25), (r2,r3) 	{NoDep, 64A}
	p0 = cmp.b32::le (s2,s3).u64, 0x1 	{NoDep, 128S}
	r8 = mov.b32 0x80 	{NoDep, 64I}
	s9 = smov.b32 0x0 	{NoDep, 64I}
	r9 = mov.b32 0x0 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r4,r5) * 1], r8 	{$11, int@2, 128C}
	send.ugm.st.d32.a64 [(r2,r3) * 1], r9 	{$12, int@1, 128C}
	sync.none 	{all, 64E}
	(p0) goto LBB3_5, LBB3_5 	{NoDep, 128B}
// %bb.3:
	s4 = smov.b32 0x1 	{NoDep, 64I}
	r2 = mov.b32 s9 	{$12.src, 64I}
	r3 = mov.b32 s9 	{NoDep, 64I}
	r4 = mov.b32 s4 	{scl@1, $11.src, 128R}
	sync.none 	{all, 64E}
LBB3_4:                                 // =>This Inner Loop Header: Depth=1
	(r0,r1) = add.s64 (r0,r1), 0x1 	{NoDep, 64A}
	(r8,r9) = shl.b64 (r0,r1), 0x2 	{int@1, $14.src, 128A}
	s4 = redfirst.b32 r4, 0xffffffff 	{NoDep, 128F}
	s10 = redfirst.b32 r3, 0xffffffff 	{NoDep, 128F}
	(r10,r11) = add.u64 (s26,s27), (r8,r9) 	{int@1, $13.src, 128A}
	s8 = sadd.s32 s4, 0x1 	{shfl@2, 64A}
	(r8,r9) = add.u64 (s24,s25), (r8,r9) 	{NoDep, 64A}
	s4 = sadd.s32 s10, 0x81 	{shfl@1, 64A}
	s10 = sadd.s32 s10, 0x1 	{NoDep, 64A}
	p0 = cmp.b32::gt (s2,s3).u64, (s8,s9) 	{scl@3, 64B}
	r5 = mov.b32 s4 	{scl@2, 64I}
	r3 = mov.b32 s10 	{scl@1, 64I}
	r4 = mov.b32 s8 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r10,r11) * 1], r5 	{$13, int@3, 128C}
	send.ugm.st.d32.a64 [(r8,r9) * 1], r2 	{$14, NoDep, 128C}
	sync.none 	{all, 64E}
	(p0) goto::b LBB3_5, LBB3_4 	{NoDep, 128B}
LBB3_5:
	join LBB3_5 	{NoDep, 128B}
	send.slm.fence.tg 	{$15, NoDep, 128C}
	sync.none 	{$15.dst, 64E}
	send.ugm.fence.tg 	{$16, NoDep, 128C}
	sync.none 	{$16.dst, 64E}
	send.gtwy.bar s0 	{$17, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.gtwy.eot s0 	{$18, NoDep, 128C}
Lfunc_end3:                             // -- Function level Xe statistics
                                        // VRT: NumGRFs=16, NumSRFs=32
                                        // NumInsts: 91
                                        // NumCompactInsts: 44
                                        // NumSyncInsts: 19
                                        // NumScalarInsts: 11
                                        // NumSRF2GRFCopies: 8
                                        // NumGRF2SRFCopies: 2
                                        // NumGRFSpillInsts: 0
                                        // NumGRFFillInsts: 0
                                        // NumSatInsts: 0
                                        // MaxGRFUsed: 12
                                        // MaxSRFUsed: 30
                                        // NumExtendedSRFLiveRange: 4
                                        // NumCycles: 268
                                        // GRFBypassEntriesPerThread: 0
	.size	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE9_clES5_EUlNS3_7nd_itemILi3EEEE_, Lfunc_end3-_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE9_clES5_EUlNS3_7nd_itemILi3EEEE_
                                        // -- End function
	.section	.text._ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE12_clES5_EUlNS3_7nd_itemILi3EEEE_,"ax",@progbits
	.type	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE12_clES5_EUlNS3_7nd_itemILi3EEEE_,@function // -- Begin function _ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE12_clES5_EUlNS3_7nd_itemILi3EEEE_
_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE12_clES5_EUlNS3_7nd_itemILi3EEEE_: // @_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE12_clES5_EUlNS3_7nd_itemILi3EEEE_
// %bb.0:
	r7 = mov.b32 0x80 	{NoDep, 64I}    // Init SP in kernel
	r6 = mov.b32 0x0 	{NoDep, 64I}    // Init FP in kernel
	gctrl = sbfia.(2,6).b32 gctrl, 0x3 	{NoDep, 128G}
	gctrl = sbfia.(1,10).b32 gctrl, 0x1 	{NoDep, 128G}
	gctrl = sbfia.(1,30).b32 gctrl, 0x1 	{NoDep, 128G}
	(s2,s3) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 160] 	{$0, NoDep, 128C}
	(s10,s11) = sasr.s64 (s2,s3), 0x3f 	{$0.dst, 128A}
	s8 = sbfn2.(s0^s1).b32 s2, s10 	{scl@1, 64D}
	s9 = sbfn2.(s0^s1).b32 s3, s11 	{NoDep, 64D}
	(s13,s14) = sadd.s64 (s8,s9), -(s10,s11) 	{scl@1, 64A}
	s31 = sasr.s32 s30, 0x1f 	{NoDep, 128A}
	s4 = sshl.b32 s14, 0xc 	{scl@2, 128A}
	s8 = sshr.b32 s13, 0x14 	{NoDep, 128A}
	(s32,s33) = sasr.s64 (s30,s31), 0x3f 	{scl@3, 128A}
	s4 = sbfn3.((s0|s1)&s2).b32 s8, s4, 0xfffff 	{scl@2, 128E}
	s8 = sbfn3.(s0&s1).b32 s13, 0xfffff, null 	{NoDep, 128E}
	s9 = sshr.b32 s14, 0x8 	{NoDep, 128A}
	r2 = cvt.f32 s4.u32 	{scl@3, 128O}
	r3 = cvt.f32 s8.u32 	{scl@2, 128O}
	s8 = sbfn2.(s0^s1).b32 s30, s32 	{f32@1, 64D}
	r4 = cvt.f32 s9.u32 	{scl@2, 128O}
	s9 = sbfn2.(s0^s1).b32 s31, s33 	{f32@1, 64D}
	r5 = mad.f32 r2, 0x49800000, r3 	{NoDep, 128A}
	r5 = mad.f32 r4, 0x53800000, r5 	{f32@1, 128A}
	(s8,s9) = sadd.s64 (s8,s9), -(s32,s33) 	{scl@1, 64A}
	r5 = eminv.f32 r5 	{f32@1, 128A}
	s4 = sshr.b32 s9, 0x8 	{scl@1, 128A}
	r8 = cvt.f32 s4.u32 	{scl@1, 128O}
	r5 = mad.f32 r5, 0xb4e00000, r5 	{em@1, 128A}
	r9 = mul.f32 r8, r5 	{f32@1, 64A}
	s4 = sshl.b32 s9, 0xc 	{NoDep, 128A}
	s9 = sshr.b32 s8, 0x14 	{NoDep, 128A}
	r9 = rnd.f32::rd r9 	{f32@1, 128A}
	r10 = mul.f32 r2, 0xc9800000 	{NoDep, 128A}
	s4 = sbfn3.((s0|s1)&s2).b32 s9, s4, 0xfffff 	{scl@1, 128E}
	r8 = mad.f32 -r3, r9, r8 	{f32@2, 128A}
	r8 = mad.f32 r10, r9, r8 	{f32@1, 64C}
	r10 = cvt.f32 s4.u32 	{scl@1, 128O}
	r11 = mad.f32 r8, 0x49800000, r10 	{f32@1, 128A}
	r11 = mul.f32 r11, r5 	{f32@1, 64A}
	r11 = rnd.f32::rd r11 	{f32@1, 128A}
	r12 = mul.f32 r3, 0x35800000 	{NoDep, 128A}
	r13 = mul.f32 r11, r12 	{f32@1, 64A}
	r13 = rnd.f32::rd r13 	{f32@1, 128A}
	r14 = mul.f32 r4, 0xc9800000 	{NoDep, 128A}
	r8 = mad.f32 -r2, r11, r8 	{NoDep, 128A}
	s4 = sbfn3.(s0&s1).b32 s8, 0xfffff, null 	{NoDep, 128E}
	r15 = mad.f32 r11, r12, -r13 	{f32@3, 128A}
	r8 = mad.f32 r14, r11, r8 	{f32@2, 64C}
	r10 = mad.f32 r15, 0xc9800000, r10 	{f32@2, 128A}
	r14 = cvt.f32 s4.u32 	{scl@1, 128O}
	r8 = mad.f32 r13, 0xbf800000, r8 	{f32@3, 128A}
	r13 = mad.f32 r10, 0x49800000, r14 	{f32@2, 128A}
	r13 = mad.f32 r8, 0x53800000, r13 	{f32@1, 128A}
	r13 = mul.f32 r13, r5 	{f32@1, 64A}
	r13 = rnd.f32::rd r13 	{f32@1, 128A}
	r15 = mul.f32 r2, 0x35800000 	{NoDep, 128A}
	r16 = mul.f32 r13, r12 	{f32@2, 64A}
	r17 = mul.f32 r13, r15 	{f32@2, 64A}
	r16 = rnd.f32::rd r16 	{f32@2, 128A}
	r17 = rnd.f32::rd r17 	{f32@2, 128A}
	r10 = mad.f32 r16, 0xbf800000, r10 	{f32@2, 128A}
	r8 = mad.f32 -r4, r13, r8 	{NoDep, 128A}
	r15 = mad.f32 r13, r15, -r17 	{f32@3, 128A}
	r12 = mad.f32 r13, r12, -r16 	{NoDep, 128A}
	r8 = mad.f32 r17, 0xbf800000, r8 	{f32@3, 128A}
	r10 = mad.f32 r15, 0xc9800000, r10 	{f32@3, 128A}
	r12 = mad.f32 r12, 0xc9800000, r14 	{f32@3, 128A}
	r14 = mad.f32 r8, 0x49800000, r10 	{f32@2, 128A}
	r14 = mad.f32 r14, 0x49800000, r12 	{f32@1, 128A}
	r5 = mul.f32 r14, r5 	{f32@1, 64A}
	s8 = sbfn2.(s0^s1).b32 s2, s10 	{NoDep, 64D}
	s9 = sbfn2.(s0^s1).b32 s3, s11 	{NoDep, 64D}
	r5 = rnd.f32::rd r5 	{f32@1, 128A}
	(s34,s35) = sadd.s64 (s8,s9), -(s10,s11) 	{scl@1, 64A}
	r2 = mad.f32 -r5, r2, r10 	{f32@1, 128A}
	r4 = mad.f32 -r5, r4, r8 	{NoDep, 128A}
	r2 = cvt.s32 r2.f32 	{f32@2, 128O}
	(r14,r15) = cvt.u64 r4.f32 	{f32@1, 128O}
	s4 = sshl.b32 s35, 0xc 	{scl@1, 128A}
	s8 = sshr.b32 s34, 0x14 	{NoDep, 128A}
	s9 = redfirst.b32 r2, 0xffffffff 	{int@2, 128F}
	r3 = mad.f32 -r5, r3, r12 	{NoDep, 128A}
	s9 = sasr.s32 s9, 0x1f 	{shfl@1, 128A}
	s4 = sbfn3.((s0|s1)&s2).b32 s8, s4, 0xfffff 	{scl@2, 128E}
	s8 = redfirst.b32 r2, 0xffffffff 	{scl@1, 128F}
	s15 = sbfn3.(s0&s1).b32 s34, 0xfffff, null 	{NoDep, 128E}
	r2 = cvt.s32 r3.f32 	{f32@1, shfl@1, 128O}
	s36 = sshr.b32 s35, 0x8 	{NoDep, 128A}
	r3 = cvt.f32 s4.u32 	{int@1, 128O}
	r4 = cvt.f32 s15.u32 	{scl@2, 128O}
	s4 = redfirst.b32 r2, 0xffffffff 	{f32@2, 128F}
	r8 = cvt.f32 s36.u32 	{scl@1, 128O}
	r10 = mad.f32 r3, 0x49800000, r4 	{f32@2, 128A}
	s37 = sasr.s32 s4, 0x1f 	{shfl@1, 128A}
	s36 = redfirst.b32 r2, 0xffffffff 	{f32@2, 128F}
	(s8,s9) = sshl.b64 (s8,s9), 0x14 	{NoDep, 128A}
	r2 = mad.f32 r8, 0x53800000, r10 	{f32@1, shfl@1, 128A}
	(s38,s39) = redfirst.b64 (r14,r15), 0xffffffff 	{NoDep, 128F}
	r2 = eminv.f32 r2 	{f32@1, 128A}
	(s8,s9) = sadd.s64 (s8,s9), (s36,s37) 	{scl@1, 64A}
	(s36,s37) = sshl.b64 (s38,s39), 0x28 	{shfl@1, 128A}
	r10 = cvt.s32 r13.f32 	{NoDep, 128O}
	r5 = cvt.s32 r5.f32 	{NoDep, 128O}
	r9 = cvt.s32 r9.f32 	{NoDep, 128O}
	s4 = redfirst.b32 r10, 0xffffffff 	{int@3, 128F}
	s15 = redfirst.b32 r5, 0xffffffff 	{int@2, 128F}
	s38 = sasr.s32 s2, 0x1f 	{NoDep, 128A}
	(s36,s37) = sadd.s64 (s36,s37), (s8,s9) 	{scl@2, 64A}
	r5 = cvt.s32 r11.f32 	{shfl@1, 128O}
	s9 = smov.b32 0x0 	{NoDep, 64I}
	s8 = redfirst.b32 r9, 0xffffffff 	{int@2, scl@2, 128F}
	s4 = sadd.s32 s4, s15 	{NoDep, 64A}
	s15 = sadd.s32 s2, s38 	{NoDep, 64A}
	s39 = redfirst.b32 r5, 0xffffffff 	{int@1, 128F}
	p0 = cmp.b32::ge (s36,s37).u64, (s13,s14) 	{NoDep, 64B}
	s13 = sadd.s32 s4, 0x1 	{int@1, scl@2, 128A}
	s14 = sbfn2.(s0^s1).b32 s15, s38 	{scl@2, 64D}
	s40 = smov.b32 s9 	{NoDep, 64I}
	(s36,s37) = sshl.b64 (s8,s9), 0x28 	{shfl@2, 128A}
	s8 = ssel.b32 s13, s4, p0 	{scl@4, 128J}
	r5 = cvt.f32 s14.u32 	{scl@4, shfl@1, 128O}
	r5 = eminv.f32 r5 	{f32@1, 128A}
	s4 = sshr.b32 s9, 0x8 	{NoDep, 128A}
	r9 = cvt.f32 s4.u32 	{scl@1, 128O}
	s4 = smov.b32 0x20 	{f32@1, 64I}
	r2 = mad.f32 r2, 0xb4e00000, r2 	{em@2, 128A}
	r10 = mul.f32 r9, r2 	{f32@1, 64A}
	s13 = sshl.b32 s9, 0xc 	{NoDep, 128A}
	s15 = sshr.b32 s4, 0x14 	{scl@2, 128A}
	r10 = rnd.f32::rd r10 	{f32@1, 128A}
	r11 = mul.f32 r3, 0xc9800000 	{NoDep, 128A}
	s13 = sbfn3.((s0|s1)&s2).b32 s15, s13, 0xfffff 	{scl@1, 128E}
	r9 = mad.f32 -r4, r10, r9 	{f32@2, 128A}
	r9 = mad.f32 r11, r10, r9 	{f32@1, 64C}
	r11 = cvt.f32 s13.u32 	{scl@1, 128O}
	r12 = mad.f32 r9, 0x49800000, r11 	{f32@1, 128A}
	r12 = mul.f32 r12, r2 	{f32@1, 64A}
	r12 = rnd.f32::rd r12 	{f32@1, 128A}
	r13 = mul.f32 r4, 0x35800000 	{NoDep, 128A}
	r14 = mul.f32 r12, r13 	{f32@1, 64A}
	r14 = rnd.f32::rd r14 	{f32@1, 128A}
	r15 = mul.f32 r8, 0xc9800000 	{NoDep, 128A}
	r9 = mad.f32 -r3, r12, r9 	{NoDep, 128A}
	s4 = sbfn3.(s0&s1).b32 s4, 0xfffff, null 	{NoDep, 128E}
	r16 = mad.f32 r12, r13, -r14 	{f32@3, 128A}
	r9 = mad.f32 r15, r12, r9 	{f32@2, 64C}
	r11 = mad.f32 r16, 0xc9800000, r11 	{f32@2, 128A}
	r15 = cvt.f32 s4.u32 	{scl@1, 128O}
	r9 = mad.f32 r14, 0xbf800000, r9 	{f32@3, 128A}
	r14 = mad.f32 r11, 0x49800000, r15 	{f32@2, 128A}
	r14 = mad.f32 r9, 0x53800000, r14 	{f32@1, 128A}
	r14 = mul.f32 r14, r2 	{f32@1, 64A}
	r14 = rnd.f32::rd r14 	{f32@1, 128A}
	r16 = mul.f32 r3, 0x35800000 	{NoDep, 128A}
	r17 = mul.f32 r14, r13 	{f32@2, 64A}
	r18 = mul.f32 r14, r16 	{f32@2, 64A}
	r17 = rnd.f32::rd r17 	{f32@2, 128A}
	r18 = rnd.f32::rd r18 	{f32@2, 128A}
	r11 = mad.f32 r17, 0xbf800000, r11 	{f32@2, 128A}
	r9 = mad.f32 -r8, r14, r9 	{NoDep, 128A}
	r16 = mad.f32 r14, r16, -r18 	{f32@3, 128A}
	r13 = mad.f32 r14, r13, -r17 	{NoDep, 128A}
	r9 = mad.f32 r18, 0xbf800000, r9 	{f32@3, 128A}
	r11 = mad.f32 r16, 0xc9800000, r11 	{f32@3, 128A}
	r13 = mad.f32 r13, 0xc9800000, r15 	{f32@3, 128A}
	r15 = mad.f32 r9, 0x49800000, r11 	{f32@2, 128A}
	r15 = mad.f32 r15, 0x49800000, r13 	{f32@1, 128A}
	r2 = mul.f32 r15, r2 	{f32@1, 64A}
	r2 = rnd.f32::rd r2 	{f32@1, 128A}
	r8 = mad.f32 -r2, r8, r9 	{f32@1, 128A}
	(r8,r9) = cvt.u64 r8.f32 	{f32@1, 128O}
	r3 = mad.f32 -r2, r3, r11 	{NoDep, 128A}
	(s39,s40) = sshl.b64 (s39,s40), 0x14 	{NoDep, 128A}
	r4 = mad.f32 -r2, r4, r13 	{NoDep, 128A}
	r3 = cvt.s32 r3.f32 	{f32@2, 128O}
	r4 = cvt.s32 r4.f32 	{f32@1, 128O}
	s4 = redfirst.b32 r3, 0xffffffff 	{int@2, 128F}
	(s36,s37) = sadd.s64 (s36,s37), (s39,s40) 	{scl@1, 64A}
	s13 = redfirst.b32 r4, 0xffffffff 	{int@1, 128F}
	s40 = sasr.s32 s4, 0x1f 	{shfl@2, 128A}
	s39 = redfirst.b32 r3, 0xffffffff 	{scl@2, 128F}
	s42 = sasr.s32 s13, 0x1f 	{shfl@2, 128A}
	s41 = redfirst.b32 r4, 0xffffffff 	{NoDep, 128F}
	(s43,s44) = redfirst.b64 (r8,r9), 0xffffffff 	{NoDep, 128F}
	(s36,s37) = sadd.s64 (s36,s37), (s8,s9) 	{NoDep, 64A}
	(s39,s40) = sshl.b64 (s39,s40), 0x14 	{scl@3, shfl@3, 128A}
	r3 = cvt.s32 r10.f32 	{NoDep, 128O}
	s45 = sbfn2.(s0^s1).b32 s9, s10 	{NoDep, 64D}
	r4 = cvt.s32 r12.f32 	{shfl@2, 128O}
	s46 = sbfn2.(s0^s1).b32 s9, s11 	{NoDep, 64D}
	s8 = redfirst.b32 r3, 0xffffffff 	{int@2, scl@4, 128F}
	s47 = redfirst.b32 r4, 0xffffffff 	{int@1, 128F}
	(s39,s40) = sadd.s64 (s39,s40), (s41,s42) 	{scl@3, 64A}
	(s41,s42) = sshl.b64 (s43,s44), 0x28 	{shfl@3, 128A}
	s48 = smov.b32 s9 	{NoDep, 64I}
	r3 = cvt.s32 r14.f32 	{shfl@2, 128O}
	r2 = cvt.s32 r2.f32 	{NoDep, 128O}
	s4 = redfirst.b32 r3, 0xffffffff 	{int@2, 128F}
	(s39,s40) = sadd.s64 (s41,s42), (s39,s40) 	{scl@2, 64A}
	s13 = redfirst.b32 r2, 0xffffffff 	{int@1, 128F}
	(s41,s42) = sshl.b64 (s8,s9), 0x28 	{NoDep, 128A}
	(s43,s44) = sshl.b64 (s47,s48), 0x14 	{scl@3, shfl@3, 128A}
	s4 = sadd.s32 s4, s13 	{shfl@1, 64A}
	r2 = mul.f32 r5, 0x4f7ffffe 	{em@1, 128A}
	p0 = cmp.b32::ge (s39,s40).u64, (s34,s35) 	{scl@4, 64B}
	s8 = sadd.s32 s4, 0x1 	{scl@1, 64A}
	(s34,s35) = sadd.s64 (s41,s42), (s43,s44) 	{int@1, 64A}
	r2 = cvt.u32 r2.f32 	{f32@1, 128O}
	s8 = ssel.b32 s8, s4, p0 	{scl@2, 128J}
	s4 = redfirst.b32 r2, 0xffffffff 	{int@1, scl@1, 128F}
	s13 = smul.s32 -s14, s4 	{shfl@1, 64A}
	(s34,s35) = sadd.s64 (s34,s35), (s8,s9) 	{NoDep, 64A}
	r2 = asr.s32 r0, 0x1f 	{NoDep, 128A}
	(s39,s40) = smullh.u64 s4, s13 	{scl@2, 128A}
	r3 = add.s32 r0, r2 	{int@1, 64A}
	s34 = sbfn2.(s0^s1).b32 s34, s45 	{scl@2, 64D}
	s35 = sbfn2.(s0^s1).b32 s35, s46 	{NoDep, 64D}
	s4 = sadd.s32 s4, s40 	{scl@3, 64A}
	r3 = bfn2.(s0^s1).b32 r3, r2 	{int@1, 64D}
	(s34,s35) = sadd.s64 (s34,s35), -(s45,s46) 	{scl@2, 64A}
	(r4,r5) = mullh.u64 r3, s4 	{int@1, scl@2, 128A}
	r4 = mul.s32 r5, s14 	{int@1, 64A}
	(r8,r9) = max.s64 (s34,s35), 0x1 	{scl@1, 128A}
	r3 = add.s32 r3, -r4 	{int@2, 64A}
	s4 = redfirst.b32 r8, 0xffffffff 	{int@2, 128F}
	p0 = cmp.b32::ge r3.u32, s14 	{int@1, 64B}
	r4 = add.s32 r3, -s14 	{NoDep, 64A}
	r10 = add.s32 r5, 0x1 	{NoDep, 64A}
	(s34,s35) = smullh.u64 s4, 0x6 	{shfl@1, 128A}
	r3 = sel.b32 r4, r3, p0 	{int@2, 128J}
	r4 = sel.b32 r10, r5, p0 	{int@2, 128J}
	s4 = redfirst.b32 r9, 0xffffffff 	{scl@1, 128F}
	p0 = cmp.b32::ge r3.u32, s14 	{int@2, 64B}
	s8 = sbfn3.(s0&s1).b32 s34, 0xfffffffe, null 	{NoDep, 128E}
	r3 = add.s32 r4, 0x1 	{int@2, 64A}
	s10 = sbfn2.(s0^s1).b32 s32, s10 	{NoDep, 64D}
	(r10,r11) = mullh.u64 r8, r1 	{NoDep, 128A}
	s11 = sbfn2.(s0^s1).b32 s33, s11 	{NoDep, 64D}
	s4 = smad.u32 s4, 0x6, s35 	{shfl@1, 128A}
	r3 = sel.b32 r3, r4, p0 	{int@2, 128J}
	(s13,s14) = smullh.u64 s8, s5 	{scl@4, 128A}
	r2 = bfn2.(s0^s1).b32 r2, s38 	{NoDep, 64D}
	s32 = sbfn2.(s0^s1).b32 s36, s10 	{scl@4, 64D}
	r4 = mov.b32 0x0 	{NoDep, 64I}
	s33 = sbfn2.(s0^s1).b32 s37, s11 	{scl@4, 64D}
	r3 = bfn2.(s0^s1).b32 r3, r2 	{int@2, 64D}
	s4 = sbfn3.(s0&s1).b32 s4, 0x0, null 	{scl@4, 128E}
	s8 = smad.u32 s8, s9, s14 	{scl@4, 64C}
	r4 = mad.u32 r8, r4, r11 	{int@2, 64C}
	r8 = add.s32 r3, -r2 	{int@2, 64A}
	(s10,s11) = sadd.s64 (s32,s33), -(s10,s11) 	{scl@3, 64A}
	r11 = mad.u32 r9, r1, r4 	{int@2, 64C}
	s14 = smad.u32 s4, s5, s8 	{scl@2, 64C}
	r9 = asr.s32 r8, 0x1f 	{int@2, 128A}
	(r2,r3) = add3.s64 (r10,r11), (s13,s14), (r8,r9) 	{int@1, scl@1, 128A}
	p0 = cmp.b32::gt s10.s32, 0x0 	{NoDep, 128S}
	s4 = smov.b32 0xffffffff 	{NoDep, 128R}
	r1, p1 = cmp.b32::lt (r2,r3).s64, (s28,s29) 	{int@2, 64B}
	s4 = ssel.b32 s4, 0x0, p0 	{scl@1, 128J}
	r4 = bfn3.(s0&s1&s2).b32 r1, s4, 0x1 	{int@1, scl@1, 128E}
	r1 = mul.s32 r8, s2 	{NoDep, 64A}
	r5 = bfn3.(~s0&s1).b32 r4, 0x1, null 	{int@2, 128E}
	r4 = add.s32 r0, -r1 	{int@2, 64A}
	s4 = smov.b32 0x0 	{NoDep, 64I}
	r5, p0 = bfn3.(s0&s1).b32::ne r5, 0x1, null 	{int@2, 128E}
	r8 = mul.s32 r4, s10 	{int@2, 64A}
	sync.none 	{all, 64E}
	(p0) goto LBB4_7, LBB4_7 	{NoDep, 128B}
// %bb.1:
	r5 = mov.b32 s30 	{NoDep, 64I}
	(r10,r11) = mullh.u64 r2, r5 	{int@1, 128A}
	r9 = mov.b32 s31 	{NoDep, 64I}
	r9 = mad.u32 r2, r9, r11 	{int@1, 64C}
	r11 = mad.u32 r3, r5, r9 	{int@1, 64C}
	r9 = asr.s32 r8, 0x1f 	{NoDep, 128A}
	(r12,r13) = shl.b64 (r10,r11), 0x1 	{int@2, 128A}
	(r10,r11) = shl.b64 (r8,r9), 0x1 	{int@2, 128A}
	s12, p1 = sbfn3.(s0&s1).b32::eq s10, 0x3, null 	{NoDep, 128E}
	p2 = cmp.b32::lt s10.u32, 0x4 	{NoDep, 128S}
	r9 = mov.b32 s4 	{NoDep, 64I}
	sync.none 	{all, 64E}
	(p2) goto LBB4_4, LBB4_4 	{NoDep, 128B}
// %bb.2:
	(r14,r15) = add.u64 (s20,s21), (r12,r13) 	{NoDep, 64A}
	(r14,r15) = add.u64 (r14,r15), (r10,r11) 	{int@1, 64A}
	s13 = sbfn3.(s0&s1).b32 s10, 0x7ffffffc, null 	{NoDep, 128E}
	(r16,r17) = add.u64 (s22,s23), (r10,r11) 	{NoDep, 64A}
	s14 = smov.b32 0x0 	{NoDep, 64I}
	s15 = smov.b32 0x40 	{NoDep, 64I}
	r5 = mov.b32 s4 	{NoDep, 64I}
	sync.none 	{all, 64E}
LBB4_3:                                 // =>This Inner Loop Header: Depth=1
	s31 = redfirst.b32 r5, 0xffffffff 	{NoDep, 128F}
	s8 = sshr.b32 s31, 0x1 	{shfl@1, 128A}
	(s32,s33) = sshl.b64 (s8,s9), 0x2 	{scl@1, 128A}
	(r18,r19) = add.u64 (r14,r15), (s32,s33) 	{scl@1, $14.src, 128A}
	s8 = sshl.b32 s8, 0x2 	{NoDep, 128A}
	(r20,r21) = send.ugm.ld.d32v2.a64 [(r18,r19) * 1] 	{$1, int@1, 128C}
	s34 = sadd.u32 s14, s8 	{scl@1, 64A}
	sync.none 	{$12.src, 64E}
	r5 = mov.b32 s34 	{scl@1, $9.src, 128R}
	r9 = send.ugm.ldpriv.d32 [(s18,s19)][r5] 	{$2, int@1, 128C}
	(r22,r23) = add.u64 (r16,r17), (s32,s33) 	{NoDep, 64A}
	sync.none 	{$1.src, 64E}
	(r18,r19) = send.ugm.ld.d32v2.a64 [(r22,r23) * 1] 	{$3, int@1, 128C}
	r24 = bfn3.(s0&s1).b32 r20, 0xffff, null 	{$1.dst, $10.src, 128E}
	s8 = sadd.u32 s15, s8 	{NoDep, 64A}
	r9 = bfn3.(s0|s1&s2).b32 r24, r9, 0xffff0000 	{int@1, $2.dst, 128E}
	r24 = mov.b32 s8 	{scl@1, 64I}
	send.ugm.stpriv.d32 [(s18,s19)][r5], r9 	{$4, int@2, 128C}
	sync.none 	{$3.src, 64E}
	r22 = send.ugm.ldpriv.d32 [(s18,s19)][r24] 	{$5, int@1, 128C}
	r23 = bfn3.(s0&s1).b32 r18, 0xffff, null 	{$3.dst, 128E}
	r22 = bfn3.(s0|s1&s2).b32 r23, r22, 0xffff0000 	{int@1, $5.dst, 128E}
	send.ugm.stpriv.d32 [(s18,s19)][r24], r22 	{$6, int@1, 128C}
	(r26,r27) = send.ugm.ldpriv.d32v2 [(s18,s19)][r5] 	{$7, NoDep, 128C}
	(r28,r29) = send.ugm.ldpriv.d32v2 [(s18,s19)][r24] 	{$8, NoDep, 128C}
	s8 = sbfn3.(s0|s1).b32 s31, 0x2, null 	{NoDep, 128E}
	s8 = sshr.b32 s8, 0x1 	{scl@1, 128A}
	r9 = bfn3.(s0&s1).b32 r20, 0xffff0000, null 	{$4.src, 128E}
	sync.none 	{$7.dst, 64E}
	r22 = bfn3.(s0|s1&s2).b32 r9, r26, 0xffff 	{int@1, $6.src, 128E}
	s8 = sshl.b32 s8, 0x2 	{scl@1, 128A}
	r9 = bfn3.(s0&s1).b32 r21, 0xffff, null 	{NoDep, 128E}
	r18 = bfn3.(s0&s1).b32 r18, 0xffff0000, null 	{NoDep, 128E}
	r20 = bfn3.(s0&s1).b32 r19, 0xffff, null 	{NoDep, 128E}
	s32 = sadd.u32 s14, s8 	{scl@1, 64A}
	r23 = bfn3.(s0|s1&s2).b32 r9, r27, 0xffff0000 	{int@3, 128E}
	r26 = bfn3.(s0|s1&s2).b32 r18, r28, 0xffff 	{int@3, $8.dst, 128E}
	r27 = bfn3.(s0|s1&s2).b32 r20, r29, 0xffff0000 	{int@3, 128E}
	r9 = mov.b32 s32 	{scl@1, 64I}
	send.ugm.stpriv.d32v2 [(s18,s19)][r5], (r22,r23) 	{$9, int@4, 128C}
	send.ugm.stpriv.d32v2 [(s18,s19)][r24], (r26,r27) 	{$10, int@2, 128C}
	r18 = send.ugm.ldpriv.d16u32 [(s18,s19)][r9] 	{$11, int@1, 128C}
	s8 = sadd.u32 s15, s8 	{NoDep, 64A}
	r5 = bfn3.(s0&s1|s2).b32 r21, 0xffff0000, r18 	{$9.src, $11.dst, 128E}
	r18 = mov.b32 s8 	{scl@1, 64I}
	send.ugm.stpriv.d32 [(s18,s19)][r9], r5 	{$12, int@2, 128C}
	r20 = send.ugm.ldpriv.d16u32 [(s18,s19)][r18] 	{$13, int@1, 128C}
	s8 = sadd.s32 s31, 0x4 	{NoDep, 64A}
	r19 = bfn3.(s0&s1|s2).b32 r19, 0xffff0000, r20 	{$13.dst, 128E}
	p2 = cmp.b32::ne s8.s32, s13 	{scl@1, 64B}
	r5 = mov.b32 s8 	{$12.src, 64I}
	r9 = mov.b32 s8 	{NoDep, 64I}
	send.ugm.stpriv.d32 [(s18,s19)][r18], r19 	{$14, int@4, 128C}
	sync.none 	{all, 64E}
	(p2) goto::b LBB4_4, LBB4_3 	{NoDep, 128B}
LBB4_4:
	join LBB4_7 	{NoDep, 128B}
	sync.none 	{all, 64E}
	(p1) goto LBB4_7, LBB4_7 	{NoDep, 128B}
// %bb.5:
	(r12,r13) = add.u64 (s20,s21), (r12,r13) 	{NoDep, 64A}
	(r12,r13) = add.u64 (r12,r13), (r10,r11) 	{int@1, 64A}
	s13 = smov.b32 0xffff 	{NoDep, 64I}
	(r10,r11) = add.u64 (s22,s23), (r10,r11) 	{NoDep, 64A}
	s14 = smov.b32 0x0 	{NoDep, 64I}
	s15 = smov.b32 0x40 	{NoDep, 64I}
	r5 = mov.b32 s4 	{$9.src, $12.src, 128R}
	sync.none 	{all, 64E}
LBB4_6:                                 // =>This Inner Loop Header: Depth=1
	s31 = redfirst.b32 r9, 0xffffffff 	{NoDep, 128F}
	s8 = sshr.b32 s31, 0x1 	{shfl@1, 128A}
	(s32,s33) = sshl.b64 (s8,s9), 0x2 	{scl@1, 128A}
	(r14,r15) = add.u64 (r12,r13), (s32,s33) 	{scl@1, $20.src, 128A}
	sync.none 	{$18.src, 64E}
	r9 = send.ugm.ld.d32.a64 [(r14,r15) * 1] 	{$15, int@1, 128C}
	s8 = sshl.b32 s8, 0x2 	{NoDep, 128A}
	s34 = sadd.u32 s14, s8 	{scl@1, 64A}
	r16 = mov.b32 s34 	{scl@1, 64I}
	r17 = send.ugm.ldpriv.d32 [(s18,s19)][r16] 	{$16, int@1, 128C}
	s34 = sshl.b32 s31, 0x4 	{NoDep, 128A}
	(r14,r15) = add.u64 (r10,r11), (s32,s33) 	{$15.src, 64A}
	s32 = sbfn3.(s0&s1).b32 s34, 0x10, null 	{int@1, scl@1, 128E}
	r18 = send.ugm.ld.d32.a64 [(r14,r15) * 1] 	{$17, $14.src, 128C}
	r9 = shr.b32 r9, s32 	{scl@1, $15.dst, 128A}
	r9 = bfn3.(s0&s1).b32 r9, 0xffff, null 	{int@1, 128E}
	s33 = sshl.b32 s13, s32 	{NoDep, 128A}
	r9 = shl.b32 r9, s32 	{int@1, 128A}
	s8 = sadd.u32 s15, s8 	{NoDep, 64A}
	sync.none 	{$16.dst, 64E}
	r9 = bfn3.(s0|s1&~s2).b32 r9, r17, s33 	{int@1, scl@2, 128E}
	r14 = mov.b32 s8 	{scl@1, $17.src, 128R}
	send.ugm.stpriv.d32 [(s18,s19)][r16], r9 	{$18, int@2, 128C}
	r15 = send.ugm.ldpriv.d32 [(s18,s19)][r14] 	{$19, int@1, 128C}
	r17 = shr.b32 r18, s32 	{$17.dst, 128A}
	r17 = bfn3.(s0&s1).b32 r17, 0xffff, null 	{int@1, 128E}
	s8 = redfirst.b32 r5, 0xffffffff 	{NoDep, 128F}
	r5 = shl.b32 r17, s32 	{int@1, shfl@1, 128A}
	s8 = sadd.s32 s8, 0x1 	{NoDep, 64A}
	s31 = sadd.s32 s31, 0x1 	{NoDep, 64A}
	r15 = bfn3.(s0|s1&~s2).b32 r5, r15, s33 	{int@1, $19.dst, 128E}
	p1 = cmp.b32::ne s8.s32, s12 	{scl@2, 64B}
	r9 = mov.b32 s31 	{scl@1, $18.src, 128R}
	r5 = mov.b32 s8 	{NoDep, 64I}
	send.ugm.stpriv.d32 [(s18,s19)][r14], r15 	{$20, int@4, 128C}
	sync.none 	{all, 64E}
	(p1) goto::b LBB4_7, LBB4_6 	{NoDep, 128B}
LBB4_7:
	join LBB4_7 	{NoDep, 128B}
	send.slm.fence.tg 	{$21, NoDep, 128C}
	sync.none 	{$21.dst, 64E}
	send.ugm.fence.tg 	{$22, NoDep, 128C}
	sync.none 	{$22.dst, 64E}
	send.gtwy.bar s0 	{$23, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	sync.none 	{all, 64E}
	sync.none 	{$23.src, 64E}
	(p0) goto LBB4_14, LBB4_14 	{NoDep, 128B}
// %bb.8:
	s8, p0 = sbfn3.(s0&s1).b32::eq s10, 0x3, null 	{NoDep, 128E}
	p1 = cmp.b32::lt s10.u32, 0x4 	{NoDep, 128S}
	r9 = mov.b32 s4 	{$12.src, $18.src, 128R}
	sync.none 	{all, 64E}
	(p1) goto LBB4_11, LBB4_11 	{NoDep, 128B}
// %bb.9:
	s12 = sbfn3.(s0&s1).b32 s10, 0x7ffffffc, null 	{NoDep, 128E}
	s13 = smov.b32 0x0 	{NoDep, 64I}
	r5 = mov.b32 s4 	{$9.src, 64I}
	sync.none 	{all, 64E}
LBB4_10:                                // =>This Inner Loop Header: Depth=1
	s14 = redfirst.b32 r5, 0xffffffff 	{NoDep, 128F}
	s15 = sshr.b32 s14, 0x1 	{shfl@1, 128A}
	s15 = sshl.b32 s15, 0x2 	{scl@1, 128A}
	s15 = sadd.u32 s13, s15 	{scl@1, 64A}
	r10 = mov.b32 s15 	{scl@1, $25.src, 128R}
	(r12,r13) = send.ugm.ldpriv.d32v2 [(s18,s19)][r10] 	{$24, int@1, 128C}
	r5 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{$24.dst, 128E}
	r5 = cvt.f32 r5.f16 	{int@1, 128O}
	r5 = mul.f32 -r5, 0x3fb8aa3b 	{f32@1, 128A}
	r5 = emexp2.f32 r5 	{f32@1, 128A}
	r9 = bfn3.(s0&s1).b32 r13, 0xffff, null 	{NoDep, 128E}
	r9 = cvt.f32 r9.f16 	{int@1, 128O}
	r9 = mul.f32 -r9, 0x3fb8aa3b 	{f32@1, 128A}
	r9 = emexp2.f32 r9 	{f32@1, 128A}
	r5 = add.f32 r5, 0x3f800000 	{em@2, 128A}
	r5 = eminv.f32 r5 	{f32@1, 128A}
	r9 = add.f32 r9, 0x3f800000 	{em@2, 128A}
	r9 = eminv.f32 r9 	{f32@1, 128A}
	r5 = cvt.f16 r5.f32 	{em@2, 128O}
	r5 = bfn3.(s0&s1|s2).b32 r12, 0xffff0000, r5 	{f32@1, 128E}
	r11 = bfe.(16,16).b32 r5 	{int@1, 128G}
	r11 = cvt.f32 r11.f16 	{int@1, 128O}
	r11 = mul.f32 -r11, 0x3fb8aa3b 	{f32@1, 128A}
	r11 = emexp2.f32 r11 	{f32@1, 128A}
	r9 = cvt.f16 r9.f32 	{em@2, 128O}
	r9 = bfn3.(s0&s1|s2).b32 r13, 0xffff0000, r9 	{f32@1, 128E}
	r12 = bfe.(16,16).b32 r9 	{int@1, 128G}
	r12 = cvt.f32 r12.f16 	{int@1, 128O}
	r12 = mul.f32 -r12, 0x3fb8aa3b 	{f32@1, 128A}
	r12 = emexp2.f32 r12 	{f32@1, 128A}
	r11 = add.f32 r11, 0x3f800000 	{em@2, 128A}
	r11 = eminv.f32 r11 	{f32@1, 128A}
	r12 = add.f32 r12, 0x3f800000 	{em@2, 128A}
	r12 = eminv.f32 r12 	{f32@1, 128A}
	r11 = cvt.f16 r11.f32 	{em@2, 128O}
	r12 = cvt.f16 r12.f32 	{em@1, 128O}
	r11 = shl.b32 r11, 0x10 	{f32@2, 128A}
	r12 = shl.b32 r12, 0x10 	{f32@1, 128A}
	s14 = sadd.s32 s14, 0x4 	{NoDep, 64A}
	r14 = bfn3.(s0|s1&s2).b32 r11, r5, 0xffff 	{int@2, $20.src, 128E}
	r15 = bfn3.(s0|s1&s2).b32 r12, r9, 0xffff 	{int@2, 128E}
	p1 = cmp.b32::ne s14.s32, s12 	{scl@1, 64B}
	r5 = mov.b32 s14 	{NoDep, 64I}
	r9 = mov.b32 s14 	{NoDep, 64I}
	send.ugm.stpriv.d32v2 [(s18,s19)][r10], (r14,r15) 	{$25, int@4, 128C}
	sync.none 	{all, 64E}
	(p1) goto::b LBB4_11, LBB4_10 	{NoDep, 128B}
LBB4_11:
	join LBB4_14 	{NoDep, 128B}
	sync.none 	{all, 64E}
	(p0) goto LBB4_14, LBB4_14 	{NoDep, 128B}
// %bb.12:
	s12 = smov.b32 0x0 	{NoDep, 64I}
	s13 = smov.b32 0xffff 	{NoDep, 64I}
	r5 = mov.b32 s4 	{$9.src, 64I}
	sync.none 	{all, 64E}
LBB4_13:                                // =>This Inner Loop Header: Depth=1
	s14 = redfirst.b32 r9, 0xffffffff 	{NoDep, 128F}
	s15 = sshr.b32 s14, 0x1 	{shfl@1, 128A}
	s15 = sshl.b32 s15, 0x2 	{scl@1, 128A}
	s15 = sadd.u32 s12, s15 	{scl@1, 64A}
	sync.none 	{$27.src, 64E}
	r10 = mov.b32 s15 	{scl@1, $25.src, 128R}
	r9 = send.ugm.ldpriv.d32 [(s18,s19)][r10] 	{$26, int@1, 128C}
	s15 = sshl.b32 s14, 0x4 	{NoDep, 128A}
	s15 = sbfn3.(s0&s1).b32 s15, 0x10, null 	{scl@1, 128E}
	r11 = shr.b32 r9, s15 	{scl@1, $26.dst, 128A}
	r11 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@1, 128E}
	r11 = cvt.f32 r11.f16 	{int@1, 128O}
	r11 = mul.f32 -r11, 0x3fb8aa3b 	{f32@1, 128A}
	r11 = emexp2.f32 r11 	{f32@1, 128A}
	r11 = add.f32 r11, 0x3f800000 	{em@1, 128A}
	r11 = eminv.f32 r11 	{f32@1, 128A}
	r11 = cvt.f16 r11.f32 	{em@1, 128O}
	s20 = redfirst.b32 r5, 0xffffffff 	{NoDep, 128F}
	s21 = sshl.b32 s13, s15 	{NoDep, 128A}
	r5 = shl.b32 r11, s15 	{f32@1, shfl@1, 128A}
	s15 = sadd.s32 s20, 0x1 	{int@1, 64A}
	s14 = sadd.s32 s14, 0x1 	{NoDep, 64A}
	r11 = bfn3.(s0|s1&~s2).b32 r5, r9, s21 	{scl@3, 128E}
	p0 = cmp.b32::ne s15.s32, s8 	{scl@2, 64B}
	r9 = mov.b32 s14 	{scl@1, 64I}
	r5 = mov.b32 s15 	{NoDep, 64I}
	send.ugm.stpriv.d32 [(s18,s19)][r10], r11 	{$27, int@4, 128C}
	sync.none 	{all, 64E}
	(p0) goto::b LBB4_14, LBB4_13 	{NoDep, 128B}
LBB4_14:
	join LBB4_14 	{NoDep, 128B}
	(s12,s13) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 176] 	{$28, NoDep, 128C}
	(s14,s15) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 184] 	{$29, NoDep, 128C}
	p0 = cmp.b32::ge (r2,r3).s64, (s28,s29) 	{NoDep, 64B}
	send.slm.fence.tg 	{$30, NoDep, 128C}
	sync.none 	{$30.dst, 64E}
	send.ugm.fence.tg 	{$31, NoDep, 128C}
	sync.none 	{$31.dst, 64E}
	send.gtwy.bar s0 	{$0, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	sync.none 	{all, 64E}
	sync.none 	{$29.src, 64E}
	sync.none 	{$0.src, 64E}
	(p0) goto LBB4_44, LBB4_44 	{NoDep, 128B}
// %bb.15:
	(s20,s21) = (W) send.ugm.ld.d64.a64 [(s16,s17) + 168] 	{$1, NoDep, 128C}
	p1 = cmp.b32::le s10.s32, 0x0 	{NoDep, 128S}
	sync.none 	{all, 64E}
	sync.none 	{$1.src, 64E}
	(p1) goto LBB4_22, LBB4_22 	{NoDep, 128B}
// %bb.16:
	s3, p2 = sbfn3.(s0&s1).b32::eq s10, 0x3, null 	{NoDep, 128E}
	p3 = cmp.b32::lt s10.u32, 0x4 	{NoDep, 128S}
	r9 = mov.b32 s4 	{$12.src, $18.src, 128R}
	sync.none 	{all, 64E}
	(p3) goto LBB4_19, LBB4_19 	{NoDep, 128B}
// %bb.17:
	s8 = sbfn3.(s0&s1).b32 s10, 0x7ffffffc, null 	{NoDep, 128E}
	s22 = smov.b32 0x0 	{NoDep, 64I}
	s23 = smov.b32 0x40 	{NoDep, 64I}
	r5 = mov.b32 s4 	{$9.src, 64I}
	sync.none 	{all, 64E}
LBB4_18:                                // =>This Inner Loop Header: Depth=1
	s28 = redfirst.b32 r5, 0xffffffff 	{NoDep, 128F}
	s29 = sshr.b32 s28, 0x1 	{shfl@1, 128A}
	s29 = sshl.b32 s29, 0x2 	{scl@1, 128A}
	s31 = sadd.u32 s23, s29 	{scl@1, 64A}
	s29 = sadd.u32 s22, s29 	{NoDep, 64A}
	sync.none 	{$4.src, $27.src, 64E}
	r10 = mov.b32 s31 	{scl@2, $25.src, 128R}
	r5 = mov.b32 s29 	{scl@1, 64I}
	(r12,r13) = send.ugm.ldpriv.d32v2 [(s18,s19)][r10] 	{$2, int@2, 128C}
	sync.none 	{$20.src, 64E}
	(r14,r15) = send.ugm.ldpriv.d32v2 [(s18,s19)][r5] 	{$3, int@1, 128C}
	r9 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{$2.dst, 128E}
	r11 = bfe.(16,0).b32 r14 	{$3.dst, 128G}
	r16 = bfn3.(s0&s1).b32 r13, 0xffff, null 	{NoDep, 128E}
	r5 = bfe.(16,0).b32 r15 	{NoDep, 128G}
	r9 = add.f16 r9, r11 	{int@3, 64A}
	r5 = add.f16 r16, r5 	{int@1, 64A}
	r9 = bfn3.(s0&s1|s2).b32 r12, 0xffff0000, r9 	{f32@2, 128E}
	r5 = bfn3.(s0&s1|s2).b32 r13, 0xffff0000, r5 	{f32@1, 128E}
	r11 = bfe.(16,16).b32 r9 	{int@2, 128G}
	r12 = bfe.(16,16).b32 r14 	{NoDep, 128G}
	r13 = bfe.(16,16).b32 r5 	{int@3, 128G}
	r14 = bfe.(16,16).b32 r15 	{NoDep, 128G}
	r11 = add.f16 r11, r12 	{int@3, 64A}
	r12 = add.f16 r13, r14 	{int@1, 64A}
	r11 = shl.b32 r11, 0x10 	{f32@2, 128A}
	r12 = shl.b32 r12, 0x10 	{f32@1, 128A}
	s28 = sadd.s32 s28, 0x4 	{NoDep, 64A}
	r14 = bfn3.(s0|s1&s2).b32 r11, r9, 0xffff 	{int@2, 128E}
	r15 = bfn3.(s0|s1&s2).b32 r12, r5, 0xffff 	{int@2, 128E}
	p3 = cmp.b32::ne s28.s32, s8 	{scl@1, 64B}
	r5 = mov.b32 s28 	{NoDep, 64I}
	r9 = mov.b32 s28 	{NoDep, 64I}
	send.ugm.stpriv.d32v2 [(s18,s19)][r10], (r14,r15) 	{$4, int@4, 128C}
	sync.none 	{all, 64E}
	(p3) goto::b LBB4_19, LBB4_18 	{NoDep, 128B}
LBB4_19:
	join LBB4_22 	{NoDep, 128B}
	sync.none 	{all, 64E}
	(p2) goto LBB4_22, LBB4_22 	{NoDep, 128B}
// %bb.20:
	s8 = smov.b32 0x0 	{NoDep, 64I}
	s22 = smov.b32 0x40 	{NoDep, 64I}
	s23 = smov.b32 0xffff 	{NoDep, 64I}
	r5 = mov.b32 s4 	{$9.src, 64I}
	sync.none 	{all, 64E}
LBB4_21:                                // =>This Inner Loop Header: Depth=1
	s28 = redfirst.b32 r9, 0xffffffff 	{NoDep, 128F}
	s29 = sshr.b32 s28, 0x1 	{shfl@1, 128A}
	s29 = sshl.b32 s29, 0x2 	{scl@1, 128A}
	s31 = sadd.u32 s8, s29 	{scl@1, 64A}
	s29 = sadd.u32 s22, s29 	{NoDep, 64A}
	r9 = mov.b32 s31 	{scl@2, 64I}
	sync.none 	{$4.src, $7.src, $27.src, 64E}
	r10 = mov.b32 s29 	{scl@1, $25.src, 128R}
	r11 = send.ugm.ldpriv.d32 [(s18,s19)][r9] 	{$5, int@2, 128C}
	r12 = send.ugm.ldpriv.d32 [(s18,s19)][r10] 	{$6, int@1, 128C}
	s29 = sshl.b32 s28, 0x4 	{NoDep, 128A}
	s29 = sbfn3.(s0&s1).b32 s29, 0x10, null 	{scl@1, 128E}
	r9 = shr.b32 r11, s29 	{scl@1, $5.dst, 128A}
	r11 = shr.b32 r12, s29 	{$6.dst, 128A}
	r9 = bfn3.(s0&s1).b32 r9, 0xffff, null 	{int@2, 128E}
	r11 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, 128E}
	r9 = add.f16 r11, r9 	{int@1, 64A}
	s31 = redfirst.b32 r5, 0xffffffff 	{NoDep, 128F}
	s32 = sshl.b32 s23, s29 	{NoDep, 128A}
	r5 = shl.b32 r9, s29 	{f32@1, shfl@1, 128A}
	s29 = sadd.s32 s31, 0x1 	{int@1, 64A}
	s28 = sadd.s32 s28, 0x1 	{NoDep, 64A}
	r11 = bfn3.(s0|s1&~s2).b32 r5, r12, s32 	{scl@3, 128E}
	p2 = cmp.b32::ne s29.s32, s3 	{scl@2, 64B}
	r9 = mov.b32 s28 	{scl@1, 64I}
	r5 = mov.b32 s29 	{NoDep, 64I}
	send.ugm.stpriv.d32 [(s18,s19)][r10], r11 	{$7, int@4, 128C}
	sync.none 	{all, 64E}
	(p2) goto::b LBB4_22, LBB4_21 	{NoDep, 128B}
LBB4_22:
	join LBB4_44 	{NoDep, 128B}
	s3 = sasr.s32 s2, 0x1f 	{NoDep, 128A}
	(s20,s21), p2 = sadd.s64::le (s2,s3), -(s20,s21) 	{scl@1, $1.dst, 128A}
	sync.none 	{all, 64E}
	(p2) goto LBB4_44, LBB4_44 	{NoDep, 128B}
// %bb.23:
	s22 = sasr.s32 s10, 0x1f 	{NoDep, 128A}
	s8 = sadd.s32 s10, s22 	{scl@1, 64A}
	s23 = sbfn2.(s0^s1).b32 s8, s22 	{scl@1, 64D}
	sync.none 	{$12.src, 64E}
	r5 = cvt.f32 s23.u32 	{scl@1, $9.src, 128O}
	r5 = eminv.f32 r5 	{f32@1, 128A}
	r9 = cvt.f32 s2.u32 	{$18.src, 128O}
	r9 = eminv.f32 r9 	{f32@1, 128A}
	s31 = smov.b32 0x7c007c00 	{NoDep, 128R}
	r5 = mul.f32 r5, 0x4f7ffffe 	{em@2, 128A}
	s32 = smov.b32 s31 	{scl@1, 64I}
	r5 = cvt.u32 r5.f32 	{f32@1, 128O}
	s8 = sshr.b32 s2, 0x1f 	{NoDep, 128A}
	s38 = redfirst.b32 r5, 0xffffffff 	{int@1, 128F}
	s33 = smov.b32 s31 	{NoDep, 64I}
	s8 = sadd.s32 s8, s2 	{scl@2, 64A}
	s36 = smul.s32 -s23, s38 	{shfl@1, 64A}
	r5 = mov.b32 lid 	{NoDep, 64I}
	s34 = smov.b32 s31 	{NoDep, 64I}
	s28 = smov.b32 0xfffffc00 	{NoDep, 128R}
	s29 = smov.b32 0xfffffc00 	{NoDep, 128R}
	s35 = sasr.s32 s8, 0x1 	{scl@5, 128A}
	(s39,s40) = smullh.u64 s38, s36 	{scl@5, 128A}
	p5 = cmp.b32::lt s10.s32, 0x1 	{NoDep, 128S}
	s8 = smov.b32 0xffffffff 	{NoDep, 128R}
	s36, p2 = sbfn3.(s0&s1).b32::eq s10, 0x7, null 	{NoDep, 128E}
	p3 = cmp.b32::lt s10.u32, 0x8 	{NoDep, 128S}
	s37 = sbfn3.(s0&s1).b32 s10, 0x7ffffff8, null 	{NoDep, 128E}
	p4 = cmp.b32::le s35.s32, 0x0 	{scl@5, 128S}
	s38 = sadd.s32 s38, s40 	{scl@4, 64A}
	r9 = mul.f32 r9, 0x4f7ffffe 	{em@1, 128A}
	s39 = ssel.b32 s8, 0x0, p5 	{scl@4, 128J}
	s40 = smov.b32 0xffff 	{NoDep, 64I}
	s41 = smov.b32 0x40 	{NoDep, 64I}
	s42 = smov.b32 0x7c00 	{NoDep, 64I}
	r9 = cvt.u32 r9.f32 	{f32@1, 128O}
	s43 = smov.b32 0x1 	{NoDep, 64I}
	sync.none 	{$4.src, $7.src, 64E}
	r10 = mov.b32 s4 	{$25.src, $27.src, 128R}
	sync.none 	{all, 64E}
LBB4_24:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB4_26 Depth 2
                                        //     Child Loop BB4_29 Depth 2
                                        //     Child Loop BB4_32 Depth 2
                                        //     Child Loop BB4_39 Depth 2
                                        //     Child Loop BB4_42 Depth 2
	r11 = mov.b32 s28 	{NoDep, 64I}
	r12 = mov.b32 s28 	{NoDep, 64I}
	sync.none 	{all, 64E}
	(p1) goto LBB4_30, LBB4_30 	{NoDep, 128B}
// %bb.25:                              //   in Loop: Header=BB4_24 Depth=1
	r15 = mov.b32 s4 	{$20.src, $10.src, 128R}
	r11 = mov.b32 s28 	{NoDep, 64I}
	r12 = mov.b32 s28 	{NoDep, 64I}
	r14 = mov.b32 s29 	{$12.src, 64I}
	r13 = mov.b32 s29 	{NoDep, 64I}
	sync.none 	{all, 64E}
	(p3) goto LBB4_27, LBB4_27 	{NoDep, 128B}
LBB4_26:                                //   Parent Loop BB4_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	s8 = redfirst.b32 r15, 0xffffffff 	{NoDep, 128F}
	s44 = sshr.b32 s8, 0x1 	{shfl@1, 128A}
	s44 = sshl.b32 s44, 0x2 	{scl@1, 128A}
	s44 = sadd.u32 s41, s44 	{scl@1, 64A}
	r15 = mov.b32 s44 	{scl@1, 64I}
	sync.none 	{$14.src, 64E}
	(r16,r17,r18,r19) = send.ugm.ldpriv.d32v4 [(s18,s19)][r15] 	{$8, int@1, 128C}
	r20 = bfe.(16,0).b32 r16 	{$8.dst, 128G}
	p5 = cmp.b32::lt r12.f16, r20 	{int@1, 64B}
	p6 = cmp.b32::lt r11.f16, r20 	{NoDep, 64B}
	r11 = sel.b32 r20, r14, p5 	{f32@1, 128J}
	r11 = sel.b32 r13, r11, p6 	{int@1, 128J}
	r12 = sel.b32 r20, r13, p6 	{NoDep, 128J}
	r13 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, 128E}
	r14 = bfe.(16,16).b32 r16 	{NoDep, 128G}
	r15 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{int@3, 128E}
	p5 = cmp.b32::lt r13.f16, r14 	{int@2, 64B}
	p6 = cmp.b32::lt r15.f16, r14 	{int@1, 64B}
	r11 = sel.b32 r14, r11, p5 	{NoDep, 128J}
	r11 = sel.b32 r12, r11, p6 	{int@1, 128J}
	r12 = sel.b32 r14, r12, p6 	{NoDep, 128J}
	r13 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, f32@2, 128E}
	r14 = bfe.(16,0).b32 r17 	{f32@1, 128G}
	r15 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{int@3, 128E}
	p5 = cmp.b32::lt r13.f16, r14 	{int@2, 64B}
	p6 = cmp.b32::lt r15.f16, r14 	{int@1, 64B}
	r11 = sel.b32 r14, r11, p5 	{NoDep, 128J}
	r11 = sel.b32 r12, r11, p6 	{int@1, 128J}
	r12 = sel.b32 r14, r12, p6 	{NoDep, 128J}
	r13 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, f32@2, 128E}
	r14 = bfe.(16,16).b32 r17 	{f32@1, 128G}
	r15 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{int@3, 128E}
	p5 = cmp.b32::lt r13.f16, r14 	{int@2, 64B}
	p6 = cmp.b32::lt r15.f16, r14 	{int@1, 64B}
	r11 = sel.b32 r14, r11, p5 	{NoDep, 128J}
	r11 = sel.b32 r12, r11, p6 	{int@1, 128J}
	r12 = sel.b32 r14, r12, p6 	{NoDep, 128J}
	r13 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, f32@2, 128E}
	r14 = bfe.(16,0).b32 r18 	{f32@1, 128G}
	r15 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{int@3, 128E}
	p5 = cmp.b32::lt r13.f16, r14 	{int@2, 64B}
	p6 = cmp.b32::lt r15.f16, r14 	{int@1, 64B}
	r11 = sel.b32 r14, r11, p5 	{NoDep, 128J}
	r11 = sel.b32 r12, r11, p6 	{int@1, 128J}
	r12 = sel.b32 r14, r12, p6 	{NoDep, 128J}
	r13 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, f32@2, 128E}
	r14 = bfe.(16,16).b32 r18 	{f32@1, 128G}
	r15 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{int@3, 128E}
	p5 = cmp.b32::lt r13.f16, r14 	{int@2, 64B}
	p6 = cmp.b32::lt r15.f16, r14 	{int@1, 64B}
	r11 = sel.b32 r14, r11, p5 	{NoDep, 128J}
	r11 = sel.b32 r12, r11, p6 	{int@1, 128J}
	r12 = sel.b32 r14, r12, p6 	{NoDep, 128J}
	r13 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, f32@2, 128E}
	r14 = bfe.(16,0).b32 r19 	{f32@1, 128G}
	r15 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{int@3, 128E}
	p5 = cmp.b32::lt r13.f16, r14 	{int@2, 64B}
	p6 = cmp.b32::lt r15.f16, r14 	{int@1, 64B}
	r11 = sel.b32 r14, r11, p5 	{NoDep, 128J}
	r11 = sel.b32 r12, r11, p6 	{int@1, 128J}
	r12 = sel.b32 r14, r12, p6 	{NoDep, 128J}
	r13 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@2, f32@2, 128E}
	r14 = bfe.(16,16).b32 r19 	{f32@1, 128G}
	r15 = bfn3.(s0&s1).b32 r12, 0xffff, null 	{int@3, 128E}
	p5 = cmp.b32::lt r13.f16, r14 	{int@2, 64B}
	p6 = cmp.b32::lt r15.f16, r14 	{int@1, 64B}
	r11 = sel.b32 r14, r11, p5 	{NoDep, 128J}
	s8 = sadd.s32 s8, 0x8 	{NoDep, 64A}
	r11 = sel.b32 r12, r11, p6 	{int@1, 128J}
	r13 = sel.b32 r14, r12, p6 	{f32@2, 128J}
	p5 = cmp.b32::ne s8.s32, s37 	{scl@1, 64B}
	r12 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@3, 128E}
	r11 = bfn3.(s0&s1).b32 r13, 0xffff, null 	{int@3, 128E}
	r15 = mov.b32 s8 	{f32@1, 64I}
	r14 = mov.b32 r12 	{int@3, 64I}
	r13 = mov.b32 r11 	{int@3, 64I}
	sync.none 	{all, 64E}
	(p5) goto::b LBB4_27, LBB4_26 	{NoDep, 128B}
LBB4_27:                                //   in Loop: Header=BB4_24 Depth=1
	join LBB4_30 	{NoDep, 128B}
	sync.none 	{all, 64E}
	(p2) goto LBB4_30, LBB4_30 	{NoDep, 128B}
// %bb.28:                              //   in Loop: Header=BB4_24 Depth=1
	r16 = mov.b32 s4 	{NoDep, 64I}
	sync.none 	{all, 64E}
LBB4_29:                                //   Parent Loop BB4_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	s8 = redfirst.b32 r15, 0xffffffff 	{NoDep, 128F}
	s44 = sshr.b32 s8, 0x1 	{shfl@1, 128A}
	s44 = sshl.b32 s44, 0x2 	{scl@1, 128A}
	s44 = sadd.u32 s41, s44 	{scl@1, 64A}
	r15 = mov.b32 s44 	{scl@1, 64I}
	r17 = send.ugm.ldpriv.d32 [(s18,s19)][r15] 	{$9, int@1, 128C}
	s44 = sshl.b32 s8, 0x4 	{NoDep, 128A}
	s44 = sbfn3.(s0&s1).b32 s44, 0x10, null 	{scl@1, 128E}
	r17 = shr.b32 r17, s44 	{scl@1, $9.dst, 128A}
	r15 = bfn3.(s0&s1).b32 r17, 0xffff, null 	{int@1, 128E}
	p5 = cmp.b32::lt r12.f16, r15 	{int@1, 64B}
	s44 = redfirst.b32 r16, 0xffffffff 	{NoDep, 128F}
	p6 = cmp.b32::lt r11.f16, r15 	{NoDep, 64B}
	r11 = sel.b32 r17, r14, p5 	{f32@1, 128J}
	s44 = sadd.s32 s44, 0x1 	{shfl@1, 64A}
	r11 = sel.b32 r13, r11, p6 	{int@1, 128J}
	r13 = sel.b32 r17, r13, p6 	{NoDep, 128J}
	s8 = sadd.s32 s8, 0x1 	{NoDep, 64A}
	p5 = cmp.b32::ne s44.s32, s36 	{scl@2, 64B}
	r12 = bfn3.(s0&s1).b32 r11, 0xffff, null 	{int@3, 128E}
	r11 = bfn3.(s0&s1).b32 r13, 0xffff, null 	{int@3, 128E}
	r15 = mov.b32 s8 	{scl@1, 64I}
	r14 = mov.b32 r12 	{int@3, 64I}
	r13 = mov.b32 r11 	{int@3, 64I}
	r16 = mov.b32 s44 	{NoDep, 64I}
	sync.none 	{all, 64E}
	(p5) goto::b LBB4_30, LBB4_29 	{NoDep, 128B}
LBB4_30:                                //   in Loop: Header=BB4_24 Depth=1
	join LBB4_44 	{NoDep, 128B}
	r13 = mov.b32 r8 	{$10.src, $12.src, 128R}
	sync.none 	{all, 64E}
	(p4) goto LBB4_36, LBB4_36 	{NoDep, 128B}
// %bb.31:                              //   in Loop: Header=BB4_24 Depth=1
	s8 = redfirst.b32 r9, 0xffffffff 	{NoDep, 128F}
	s44 = smul.s32 -s2, s8 	{shfl@1, 64A}
	(s44,s45) = smullh.u64 s8, s44 	{scl@1, 128A}
	s8 = sadd.s32 s8, s45 	{scl@1, 64A}
	(r14,r15) = mullh.u64 r5, s8 	{scl@1, $20.src, 128A}
	r13 = mul.s32 r15, s2 	{int@1, 64A}
	r13 = add.s32 r5, -r13 	{int@1, 64A}
	p5 = cmp.b32::ge r13.u32, s2 	{int@1, 64B}
	r14 = add.s32 r13, -s2 	{NoDep, 64A}
	r13 = sel.b32 r14, r13, p5 	{int@1, 128J}
	p5 = cmp.b32::ge r13.u32, s2 	{int@1, 64B}
	r14 = add.s32 r13, -s2 	{NoDep, 64A}
	r12 = add.f16 r11, r12 	{NoDep, 64A}
	r11 = sel.b32 r14, r13, p5 	{int@1, f32@1, 128J}
	r14 = mov.b32 s35 	{NoDep, 64I}
	r13 = mov.b32 r8 	{NoDep, 64I}
	sync.none 	{all, 64E}
LBB4_32:                                //   Parent Loop BB4_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	r15 = bfn2.(s0^s1).b32 r11, r14 	{NoDep, 64D}
	p5 = cmp.b32::lt r15.u32, s2 	{int@1, 64B}
	r15 = add3.s32 r5, -r11, r15 	{NoDep, 128A}
	r15 = sel.b32 r15, r5, p5 	{int@1, 128J}
	r16 = cvt.f32 r12.f16 	{NoDep, 128O}
	r17 = bfi.(13,8).b32 r15, 0x1f 	{int@1, 128G}
	r15 = shfli.b32 r16, r17, 0xffffffff 	{int@1, f32@1, 128F}
	r15 = cvt.f16 r15.f32 	{shfl@1, 128O}
	r16 = shfli.b32 r13, r17, 0xffffffff 	{NoDep, 128F}
	p5 = cmp.b32::gt r12.f16, r15 	{f32@1, 64B}
	sync.none 	{all, 64E}
	(!p5) goto LBB4_34, LBB4_34 	{NoDep, 128B}
// %bb.33:                              //   in Loop: Header=BB4_32 Depth=2
	r12 = mov.b32 r15 	{NoDep, 64I}
	r13 = mov.b32 r16 	{NoDep, 64I}
	goto LBB4_34, LBB4_35 	{NoDep, 128B}
LBB4_34:                                //   in Loop: Header=BB4_32 Depth=2
	join LBB4_35 	{NoDep, 128B}
	r17, p5 = cmp.b32::eq r12.f16, r15 	{NoDep, 64B}
	r18, p5 = cmp.b32::gt r16.s32, r13 	{$14.src, 64B}
	r17 = bfn3.(s0&s1&s2).b32 r17, r18, 0x1 	{int@1, f32@1, 128E}
	r17 = bfn3.((s0|~s1)^~s2).b32 r17, s43, 0x1 	{int@1, 128E}
	r17, p5 = bfn3.(s0&s1).b32::ne r17, 0x1, null 	{int@1, 128E}
	r12 = (p5) mov.b32 r15 	{NoDep, 64I}
	r13 = (p5) mov.b32 r16 	{NoDep, 64I}
	sync.none 	{all, 64E}
LBB4_35:                                //   in Loop: Header=BB4_32 Depth=2
	join LBB4_36 	{NoDep, 128B}
	s8 = redfirst.b32 r14, 0xffffffff 	{NoDep, 128F}
	s8 = sshr.b32 s8, 0x1 	{shfl@1, 128A}
	p5 = cmp.b32::lt r14.u32, 0x2 	{NoDep, 128S}
	r14 = mov.b32 s8 	{scl@1, 64I}
	sync.none 	{all, 64E}
	(!p5) goto::b LBB4_36, LBB4_32 	{NoDep, 128B}
LBB4_36:                                //   in Loop: Header=BB4_24 Depth=1
	join LBB4_44 	{NoDep, 128B}
	r11 = asr.s32 r13, 0x1f 	{NoDep, 128A}
	r12 = add.s32 r13, r11 	{int@1, 64A}
	r12 = bfn2.(s0^s1).b32 r12, r11 	{int@1, 64D}
	(r14,r15) = mullh.u64 r12, s38 	{int@1, $20.src, 128A}
	r13 = mul.s32 r15, s23 	{int@1, 64A}
	r12 = add.s32 r12, -r13 	{int@1, 64A}
	p5 = cmp.b32::ge r12.u32, s23 	{int@1, 64B}
	r13 = add.s32 r12, -s23 	{NoDep, 64A}
	r14 = add.s32 r15, 0x1 	{NoDep, 64A}
	r12 = sel.b32 r13, r12, p5 	{int@2, 128J}
	r13 = sel.b32 r14, r15, p5 	{int@2, 128J}
	p5 = cmp.b32::ge r12.u32, s23 	{int@2, 64B}
	r12 = add.s32 r13, 0x1 	{int@2, 64A}
	r12 = sel.b32 r12, r13, p5 	{int@1, 128J}
	r11 = bfn2.(s0^s1).b32 r11, s22 	{NoDep, 64D}
	r12 = bfn2.(s0^s1).b32 r12, r11 	{int@1, 64D}
	r11 = add.s32 r12, -r11 	{int@1, 64A}
	r11, p5 = cmp.b32::ne r4.s32, r11 	{int@1, 64B}
	r11 = bfn2.(s0|s1).b32 r11, s39 	{int@1, 64D}
	r11, p5 = bfn3.(s0&s1).b32::ne r11, 0x1, null 	{int@1, 128E}
	sync.none 	{all, 64E}
	(p5) goto LBB4_43, LBB4_43 	{NoDep, 128B}
// %bb.37:                              //   in Loop: Header=BB4_24 Depth=1
	r11 = mov.b32 s4 	{NoDep, 64I}
	sync.none 	{all, 64E}
	(p3) goto LBB4_40, LBB4_40 	{NoDep, 128B}
// %bb.38:                              //   in Loop: Header=BB4_24 Depth=1
	r12 = mov.b32 s4 	{NoDep, 64I}
	sync.none 	{all, 64E}
LBB4_39:                                //   Parent Loop BB4_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	s8 = redfirst.b32 r12, 0xffffffff 	{NoDep, 128F}
	s44 = sshr.b32 s8, 0x1 	{shfl@1, 128A}
	s44 = sshl.b32 s44, 0x2 	{scl@1, 128A}
	s44 = sadd.u32 s41, s44 	{scl@1, 64A}
	s8 = sadd.s32 s8, 0x8 	{NoDep, 64A}
	r13 = mov.b32 s31 	{$10.src, 64I}
	(r14,r15) = mov.b64 (s32,s33) 	{NoDep, 64I}
	r16 = mov.b32 s34 	{NoDep, 64I}
	r17 = mov.b32 s44 	{scl@2, 64I}
	p5 = cmp.b32::ne s8.s32, s37 	{scl@1, 64B}
	r12 = mov.b32 s8 	{NoDep, 64I}
	r11 = mov.b32 s8 	{NoDep, 64I}
	send.ugm.stpriv.d32v4 [(s18,s19)][r17], (r13,r14,r15,r16) 	{$10, int@4, 128C}
	sync.none 	{all, 64E}
	(p5) goto::b LBB4_40, LBB4_39 	{NoDep, 128B}
LBB4_40:                                //   in Loop: Header=BB4_24 Depth=1
	join LBB4_43 	{NoDep, 128B}
	sync.none 	{all, 64E}
	(p2) goto LBB4_43, LBB4_43 	{NoDep, 128B}
// %bb.41:                              //   in Loop: Header=BB4_24 Depth=1
	r12 = mov.b32 s4 	{NoDep, 64I}
	sync.none 	{all, 64E}
LBB4_42:                                //   Parent Loop BB4_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	s8 = redfirst.b32 r11, 0xffffffff 	{NoDep, 128F}
	s44 = sshr.b32 s8, 0x1 	{shfl@1, 128A}
	s44 = sshl.b32 s44, 0x2 	{scl@1, 128A}
	s44 = sadd.u32 s41, s44 	{scl@1, 64A}
	sync.none 	{$12.src, 64E}
	r13 = mov.b32 s44 	{scl@1, $10.src, 128R}
	r11 = send.ugm.ldpriv.d32 [(s18,s19)][r13] 	{$11, int@1, 128C}
	s44 = sshl.b32 s8, 0x4 	{NoDep, 128A}
	s44 = sbfn3.(s0&s1).b32 s44, 0x10, null 	{scl@1, 128E}
	s45 = redfirst.b32 r12, 0xffffffff 	{NoDep, 128F}
	s46 = sshl.b32 s40, s44 	{scl@1, 128A}
	s44 = sshl.b32 s42, s44 	{NoDep, 128A}
	s45 = sadd.s32 s45, 0x1 	{shfl@1, 64A}
	s8 = sadd.s32 s8, 0x1 	{NoDep, 64A}
	r14 = bfn3.(s0&~s1|s2).b32 r11, s46, s44 	{scl@3, $11.dst, 128E}
	p5 = cmp.b32::ne s45.s32, s36 	{scl@2, 64B}
	r11 = mov.b32 s8 	{scl@1, 64I}
	r12 = mov.b32 s45 	{NoDep, 64I}
	send.ugm.stpriv.d32 [(s18,s19)][r13], r14 	{$12, int@4, 128C}
	sync.none 	{all, 64E}
	(p5) goto::b LBB4_43, LBB4_42 	{NoDep, 128B}
LBB4_43:                                //   in Loop: Header=BB4_24 Depth=1
	join LBB4_44 	{NoDep, 128B}
	s8 = redfirst.b32 r10, 0xffffffff 	{NoDep, 128F}
	s8 = sadd.s32 s8, 0x1 	{shfl@1, 64A}
	p5 = cmp.b32::le (s20,s21).s64, (s8,s9) 	{scl@1, 64B}
	r10 = mov.b32 s8 	{NoDep, 64I}
	sync.none 	{all, 64E}
	(!p5) goto::b LBB4_44, LBB4_24 	{NoDep, 128B}
LBB4_44:
	join LBB4_44 	{NoDep, 128B}
	send.slm.fence.tg 	{$13, NoDep, 128C}
	sync.none 	{$13.dst, 64E}
	send.ugm.fence.tg 	{$14, NoDep, 128C}
	sync.none 	{$14.dst, 64E}
	send.gtwy.bar s0 	{$15, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	sync.none 	{all, 64E}
	sync.none 	{$15.src, 64E}
	(p0) goto LBB4_49, LBB4_49 	{NoDep, 128B}
// %bb.45:
	s2 = smov.b32 0xffffffff 	{NoDep, 128R}
	p0 = cmp.b32::le (s14,s15).s64, 0x0 	{$29.dst, 128S}
	s2 = ssel.b32 s2, 0x0, p0 	{scl@1, 128J}
	r0, p0 = cmp.b32::ne r0.s32, r1 	{NoDep, 64B}
	sync.none 	{scl@1, 64E}
	r0 = bfn2.(s0|s1).b32 s2, r0 	{int@1, 64D}
	r0, p0 = bfn3.(s0&s1).b32::ne r0, 0x1, null 	{int@1, 128E}
	sync.none 	{all, 64E}
	(p0) goto LBB4_49, LBB4_49 	{NoDep, 128B}
// %bb.46:
	r0 = mov.b32 s12 	{$28.dst, 64I}
	sync.none 	{$12.src, 64E}
	(r4,r5) = mullh.u64 r2, r0 	{int@1, $9.src, 128A}
	r1 = mov.b32 s13 	{NoDep, 64I}
	r1 = mad.u32 r2, r1, r5 	{int@1, 64C}
	r5 = mad.u32 r3, r0, r1 	{int@1, 64C}
	(r0,r1) = add3.s64 (s12,s13), -(s14,s15), (r4,r5) 	{int@1, 128A}
	(r2,r3) = shl.b64 (r0,r1), 0x2 	{int@1, 128A}
	(r4,r5) = add.u64 (s26,s27), (r2,r3) 	{int@1, 64A}
	(r2,r3) = add.u64 (s24,s25), (r2,r3) 	{NoDep, 64A}
	p0 = cmp.b32::le (s14,s15).u64, 0x1 	{NoDep, 128S}
	r8 = mov.b32 s30 	{NoDep, 64I}
	s3 = smov.b32 0x0 	{NoDep, 64I}
	r9 = mov.b32 0x0 	{$18.src, 64I}
	send.ugm.st.d32.a64 [(r4,r5) * 1], r8 	{$16, int@2, 128C}
	send.ugm.st.d32.a64 [(r2,r3) * 1], r9 	{$17, int@1, 128C}
	sync.none 	{all, 64E}
	(p0) goto LBB4_49, LBB4_49 	{NoDep, 128B}
// %bb.47:
	s2 = smov.b32 0x1 	{NoDep, 64I}
	r2 = mov.b32 s3 	{$17.src, 64I}
	r3 = mov.b32 s4 	{NoDep, 64I}
	r4 = mov.b32 s2 	{scl@1, $16.src, 128R}
	sync.none 	{all, 64E}
LBB4_48:                                // =>This Inner Loop Header: Depth=1
	(r0,r1) = add.s64 (r0,r1), 0x1 	{NoDep, 64A}
	(r8,r9) = shl.b64 (r0,r1), 0x2 	{int@1, $19.src, 128A}
	s8 = redfirst.b32 r3, 0xffffffff 	{NoDep, 128F}
	s2 = redfirst.b32 r4, 0xffffffff 	{NoDep, 128F}
	s9 = sadd.s32 s30, s8 	{shfl@2, 64A}
	sync.none 	{$4.src, $7.src, $18.src, $27.src, 64E}
	(r10,r11) = add.u64 (s26,s27), (r8,r9) 	{int@1, $25.src, 128A}
	s2 = sadd.s32 s2, 0x1 	{shfl@1, 64A}
	(r8,r9) = add.u64 (s24,s25), (r8,r9) 	{NoDep, 64A}
	s9 = sadd.s32 s9, 0x1 	{scl@2, 64A}
	s8 = sadd.s32 s8, 0x1 	{NoDep, 64A}
	p0 = cmp.b32::gt (s14,s15).u64, (s2,s3) 	{scl@3, 64B}
	r5 = mov.b32 s9 	{scl@2, 64I}
	r3 = mov.b32 s8 	{scl@1, 64I}
	r4 = mov.b32 s2 	{NoDep, 64I}
	send.ugm.st.d32.a64 [(r10,r11) * 1], r5 	{$18, int@3, 128C}
	send.ugm.st.d32.a64 [(r8,r9) * 1], r2 	{$19, NoDep, 128C}
	sync.none 	{all, 64E}
	(p0) goto::b LBB4_49, LBB4_48 	{NoDep, 128B}
LBB4_49:
	join LBB4_49 	{NoDep, 128B}
	send.slm.fence.tg 	{$20, NoDep, 128C}
	sync.none 	{$20.dst, 64E}
	send.ugm.fence.tg 	{$21, NoDep, 128C}
	sync.none 	{$21.dst, 64E}
	send.gtwy.bar s0 	{$22, NoDep, 128C}
	sync.barid 0x0 	{NoDep, 64E}
	send.gtwy.eot s0 	{$23, NoDep, 128C}
Lfunc_end4:                             // -- Function level Xe statistics
                                        // VRT: NumGRFs=32, NumSRFs=64
                                        // NumInsts: 960
                                        // NumCompactInsts: 379
                                        // NumSyncInsts: 82
                                        // NumScalarInsts: 219
                                        // NumSRF2GRFCopies: 73
                                        // NumGRF2SRFCopies: 42
                                        // NumGRFSpillInsts: 0
                                        // NumGRFFillInsts: 0
                                        // NumSatInsts: 0
                                        // MaxGRFUsed: 30
                                        // MaxSRFUsed: 49
                                        // NumExtendedSRFLiveRange: 25
                                        // NumCycles: 2968
                                        // GRFBypassEntriesPerThread: 0
	.size	_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE12_clES5_EUlNS3_7nd_itemILi3EEEE_, Lfunc_end4-_ZTSZZ14moe_fused_gateRN2at6TensorES1_lllldENKUlRN4sycl3_V17handlerEE12_clES5_EUlNS3_7nd_itemILi3EEEE_
                                        // -- End function
	// -- Kernel level Xe Statistics
	// GRFBypassEntriesPerThread -- max = 0
	// MaxGRFUsed -- max = 30
	// MaxSRFUsed -- max = 49
	// NumCompactInsts -- sum = 555
	// NumCycles -- sum = 4040
	// NumExtendedSRFLiveRange -- sum = 41
	// NumGRF2SRFCopies -- sum = 50
	// NumGRFFillInsts -- sum = 0
	// NumGRFSpillInsts -- sum = 0
	// NumInsts -- sum = 1324
	// NumSRF2GRFCopies -- sum = 105
	// NumSatInsts -- sum = 0
	// NumScalarInsts -- sum = 263
	// NumSyncInsts -- sum = 158
