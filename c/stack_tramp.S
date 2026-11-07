/*
 * Call a function on another stack, for vocoder/py/radio_link.py: the debug command 'x' runs the call in the
 * cable task, whose 2 KB stack is far too small for the vocoder (it runs on the HR_C7000 task's 8 KB).
 * Position independent; uploaded into free RAM and called with
 *   r0 = function, r1 = top of the new stack, r2 = its first argument, r3 = its second argument.
 * Returns the function's r0. r8 (callee saved) keeps the old stack pointer across the call.
 */
	.text
	.global tramp
tramp:
	subi	sp, sp, 8
	st.w	r15, (sp, 0)
	st.w	r8, (sp, 4)
	mov	r8, sp
	mov	sp, r1
	mov	r12, r0
	mov	r0, r2
	mov	r1, r3
	jsr	r12
	mov	sp, r8
	ld.w	r8, (sp, 4)
	ld.w	r15, (sp, 0)
	addi	sp, sp, 8
	rts

/*
 * encode_n: r0 = codecEncodeBlock, r1 = top of the new stack, r2 = output (9 bytes a frame), r3 = number of frames.
 * Calls r0(r2 + 9 i) for i = 0..r3-1 on the new stack. r8..r11 (callee saved) keep the old stack pointer, the
 * function, the output pointer and the count.
 */
	.global encode_n
encode_n:
	subi	sp, sp, 20
	st.w	r15, (sp, 0)
	st.w	r8, (sp, 4)
	st.w	r9, (sp, 8)
	st.w	r10, (sp, 12)
	st.w	r11, (sp, 16)
	mov	r8, sp
	mov	sp, r1
	mov	r9, r0
	mov	r10, r2
	mov	r11, r3
1:
	cmpnei	r11, 0
	bf	2f
	mov	r0, r10
	jsr	r9
	addi	r10, r10, 9
	subi	r11, r11, 1
	br	1b
2:
	mov	sp, r8
	ld.w	r11, (sp, 16)
	ld.w	r10, (sp, 12)
	ld.w	r9, (sp, 8)
	ld.w	r8, (sp, 4)
	ld.w	r15, (sp, 0)
	addi	sp, sp, 20
	rts
