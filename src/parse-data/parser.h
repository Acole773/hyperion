#ifndef __PARSER_H
#define __PARSER_H

void rate_library_create(char* filename, int size);

void rate_library_fixup_unused_reactants(void);

void network_create(char* filename, int size);

void data_init(void);

#endif
