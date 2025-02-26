for isym = 1, 8 do
    -- 0 2 4 6
    if isym == 1 or isym == 3 or isym == 5 or isym == 7 then
        fci {
            sys = read_in {
                int_file = '../FCIDUMP',
                nel = 10,
                ms = 0,
                sym = isym - 1,
            },
            reference = {
                det = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                ex_level = 2,
            },
            fci = {
                sorted_diagonal = true,
                write_nwfns = -1,
                wfn_file = '0' .. isym .. 'ISYM.WFN',
                write_determinants = true,
                determinant_file = '0' .. isym .. 'ISYM.DET',
                write_hamiltonian = true,
                hamiltonian_file = '0' .. isym .. 'ISYM.HAM',
            },
        }
    end
end
